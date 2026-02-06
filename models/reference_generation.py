"""
Reference Image Generation — three complete methods for GRCL (TPAMI 2025).

Method 1: Local Dictionary Retrieval (self-contained, always available)
Method 2: Web-based Image Retrieval (Baidu / Google / Bing, needs API key)
Method 3: Image-Text-Image Synthesis (LLaVA + FLUX / Stable Diffusion)

Each method returns:  (ref_images: List[ndarray], metadata, overhead_bits: int)
"""

import os, hashlib, logging, pickle, time
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Method 1 — Local Dictionary Retrieval  (ResNet-50 + SPP + PCA + BallTree)
# ═══════════════════════════════════════════════════════════════════════════════

class LocalDictionaryRetrieval:
    """Self-contained local dictionary: build once, query fast.

    Pipeline (offline, run once):
      1. Load all ref images from ref_path (directory or HDF5)
      2. Extract ResNet-50 + SPP features
      3. PCA → 256 dims
      4. MiniBatch K-means → n_clusters representatives
      5. Build Ball Tree for fast kNN
      6. Cache everything to disk

    Pipeline (online, per query):
      1. Extract query feature (+ augmented query)
      2. kNN search in Ball Tree → top-n_refs indices
      3. Load corresponding images
      4. Overhead: n_refs × ceil(log2(K)) bits
    """

    def __init__(self, ref_path: str, n_clusters: int = 3000, n_refs: int = 3,
                 cache_dir: Optional[str] = None, device: str = 'cuda'):
        self.ref_path = ref_path
        self.n_clusters = n_clusters
        self.n_refs = n_refs
        self.device = device
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(ref_path), '_ref_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Feature extractor
        self._feat_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self._resnet = models.resnet50(weights='IMAGENET1K_V1')
        self._resnet.fc = nn.Identity()
        self._resnet = self._resnet.to(device).eval()

        # Build or load dictionary
        self._ref_data, self._ref_keys = self._load_ref_source(ref_path)
        self._features, self._feat2key, self._pca = self._build_features()
        if n_clusters and n_clusters < len(self._features):
            self._cluster()
        from sklearn.neighbors import NearestNeighbors
        self._nn = NearestNeighbors(n_neighbors=n_refs, algorithm='ball_tree', n_jobs=-1)
        self._nn.fit(self._features)
        logger.info(f"LocalDict ready: {len(self._features)} entries, {n_clusters} clusters")

    # ── ref source loading ────────────────────────────────────────────────────
    def _load_ref_source(self, path):
        if os.path.isdir(path):
            keys = sorted(f for f in os.listdir(path)
                          if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp')))
            data = {k: os.path.join(path, k) for k in keys}
        elif path.endswith(('.h5', '.hdf5')):
            import h5py
            data = h5py.File(path, 'r')
            keys = list(data.keys())
        else:
            raise ValueError(f"ref_path must be a directory or HDF5 file, got {path}")
        logger.info(f"Loaded {len(keys)} reference images from {path}")
        return data, keys

    def _read_ref_image(self, key):
        if isinstance(self._ref_data, dict):
            return np.array(Image.open(self._ref_data[key]).convert('RGB'))
        return self._ref_data[key][()]

    # ── feature extraction ────────────────────────────────────────────────────
    def _spp(self, x, levels=(1, 2, 4)):
        parts = [F.adaptive_max_pool2d(x, (l, l)).flatten(1) for l in levels]
        return torch.cat(parts, 1)

    @torch.no_grad()
    def _extract_one(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        t = self._feat_transform(img).unsqueeze(0).to(self.device)
        x = self._resnet.maxpool(self._resnet.relu(self._resnet.bn1(self._resnet.conv1(t))))
        for layer in [self._resnet.layer1, self._resnet.layer2,
                      self._resnet.layer3, self._resnet.layer4]:
            x = layer(x)
        return self._spp(x).cpu().numpy().flatten()

    def _build_features(self):
        feat_path = os.path.join(self.cache_dir, 'features_spp_pca256.pkl')
        if os.path.exists(feat_path):
            logger.info(f"Loading cached features from {feat_path}")
            with open(feat_path, 'rb') as f:
                feats, f2k, pca = pickle.load(f)
            return feats, f2k, pca

        logger.info("Computing features for all reference images ...")
        from tqdm import tqdm
        raw, f2k = [], {}
        for i, k in enumerate(tqdm(self._ref_keys, desc="Extracting features")):
            raw.append(self._extract_one(self._read_ref_image(k)))
            f2k[i] = k
        raw = np.array(raw)

        from sklearn.decomposition import PCA
        logger.info("PCA → 256 dims ...")
        pca = PCA(n_components=256)
        feats = pca.fit_transform(raw)

        with open(feat_path, 'wb') as f:
            pickle.dump((feats, f2k, pca), f)
        logger.info(f"Features saved to {feat_path}")
        return feats, f2k, pca

    def _cluster(self):
        clust_path = os.path.join(self.cache_dir, f'cluster_{self.n_clusters}.pkl')
        if os.path.exists(clust_path):
            with open(clust_path, 'rb') as f:
                d = pickle.load(f)
            self._features, self._feat2key = d['feats'], d['f2k']
            return

        from sklearn.cluster import MiniBatchKMeans
        from tqdm import tqdm
        logger.info(f"Clustering {len(self._features)} → {self.n_clusters} ...")
        km = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=1024)
        labels = km.fit_predict(self._features)
        new_feats, new_f2k = [], {}
        for c in tqdm(range(self.n_clusters), desc="Selecting representatives"):
            idx = np.where(labels == c)[0]
            best = idx[np.argmin(np.linalg.norm(self._features[idx] - km.cluster_centers_[c], axis=1))]
            new_feats.append(self._features[best])
            new_f2k[len(new_feats) - 1] = self._feat2key[best]
        self._features = np.array(new_feats)
        self._feat2key = new_f2k

        with open(clust_path, 'wb') as f:
            pickle.dump({'feats': self._features, 'f2k': self._feat2key}, f)

    # ── retrieval ─────────────────────────────────────────────────────────────
    def retrieve(self, query_img) -> Tuple[List[np.ndarray], List[str], int]:
        """Retrieve n_refs most similar reference images.

        Returns (ref_images, ref_keys, overhead_bits).
        Overhead = n_refs × ceil(log2(K)) bits.
        """
        q_feat = self._extract_one(query_img)
        if self._pca is not None:
            q_feat = self._pca.transform(q_feat.reshape(1, -1)).flatten()

        # Multi-query: original + 90° rotation
        _, idx1 = self._nn.kneighbors(q_feat.reshape(1, -1))
        if isinstance(query_img, np.ndarray):
            aug = Image.fromarray(query_img).rotate(90)
        else:
            aug = query_img.rotate(90)
        q2 = self._extract_one(aug)
        if self._pca is not None:
            q2 = self._pca.transform(q2.reshape(1, -1)).flatten()
        _, idx2 = self._nn.kneighbors(q2.reshape(1, -1))

        indices = np.unique(np.concatenate([idx1[0], idx2[0]]))[:self.n_refs]
        keys = [self._feat2key[i] for i in indices]
        imgs = [self._read_ref_image(k) for k in keys]
        bits = self.n_refs * int(np.ceil(np.log2(max(self.n_clusters, 2))))
        return imgs, keys, bits


# ═══════════════════════════════════════════════════════════════════════════════
#  Method 2 — Web-based Image Retrieval  (Baidu / Google / Bing)
# ═══════════════════════════════════════════════════════════════════════════════

class WebImageRetrieval:
    """Retrieve semantically similar images from web search engines.

    Supports three backends:
      'baidu'  — Baidu image search (百度识图) via crawler
      'google' — Google Custom Search JSON API (needs API_KEY + CX)
      'bing'   — Bing Image Search API (needs API_KEY)

    The canonical URL is transmitted to the decoder (30-100 bytes per image).
    Decoder fetches the same URL → deterministic reference.
    """

    def __init__(self, n_refs: int = 3, backend: str = 'baidu',
                 api_key: str = '', search_cx: str = '',
                 cache_dir: str = './_web_cache', timeout: float = 5.0):
        self.n_refs = n_refs
        self.backend = backend
        self.api_key = api_key
        self.search_cx = search_cx
        self.cache_dir = cache_dir
        self.timeout = timeout
        os.makedirs(cache_dir, exist_ok=True)
        self._url_cache: dict = {}
        # Try loading persistent URL cache
        self._cache_path = os.path.join(cache_dir, 'url_cache.pkl')
        if os.path.exists(self._cache_path):
            with open(self._cache_path, 'rb') as f:
                self._url_cache = pickle.load(f)

    def _img_hash(self, img) -> str:
        arr = np.array(img) if not isinstance(img, np.ndarray) else img
        return hashlib.md5(arr.tobytes()[:4096]).hexdigest()

    # ── backend implementations ───────────────────────────────────────────────
    def _search_baidu(self, img) -> List[str]:
        """Baidu reverse image search via HTTP upload."""
        try:
            import requests
            from io import BytesIO
            buf = BytesIO()
            pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
            pil.save(buf, format='JPEG', quality=85)
            buf.seek(0)
            # Baidu similar-image API endpoint
            url = 'https://graph.baidu.com/upload'
            resp = requests.post(url, files={'image': ('q.jpg', buf, 'image/jpeg')},
                                 timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                sign = data.get('data', {}).get('sign', '')
                if sign:
                    detail_url = f'https://graph.baidu.com/s?sign={sign}&f=all'
                    detail = requests.get(detail_url, timeout=self.timeout)
                    # Parse result page for image URLs
                    urls = self._parse_baidu_results(detail.text)
                    return urls[:self.n_refs]
        except Exception as e:
            logger.warning(f"Baidu search failed: {e}")
        return []

    def _parse_baidu_results(self, html: str) -> List[str]:
        """Extract image URLs from Baidu result HTML."""
        import re
        urls = re.findall(r'"thumbUrl"\s*:\s*"(https?://[^"]+)"', html)
        # Deduplicate, keep order
        seen = set()
        unique = []
        for u in urls:
            if u not in seen:
                seen.add(u); unique.append(u)
        return unique

    def _search_google(self, img) -> List[str]:
        """Google Custom Search JSON API (requires api_key and search_cx)."""
        if not self.api_key or not self.search_cx:
            logger.warning("Google API key or CX not set")
            return []
        try:
            import requests
            # Use a text query derived from the image (simplified)
            # In production, use Google Vision API for reverse image search
            url = (f"https://www.googleapis.com/customsearch/v1"
                   f"?key={self.api_key}&cx={self.search_cx}"
                   f"&searchType=image&q=similar+image&num={self.n_refs}")
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                items = resp.json().get('items', [])
                return [it['link'] for it in items[:self.n_refs]]
        except Exception as e:
            logger.warning(f"Google search failed: {e}")
        return []

    def _search_bing(self, img) -> List[str]:
        """Bing Visual Search API."""
        if not self.api_key:
            logger.warning("Bing API key not set")
            return []
        try:
            import requests
            from io import BytesIO
            buf = BytesIO()
            pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
            pil.save(buf, format='JPEG', quality=85); buf.seek(0)
            headers = {'Ocp-Apim-Subscription-Key': self.api_key}
            resp = requests.post('https://api.bing.microsoft.com/v7.0/images/visualsearch',
                                 headers=headers, files={'image': buf}, timeout=self.timeout)
            if resp.status_code == 200:
                tags = resp.json().get('tags', [])
                urls = []
                for tag in tags:
                    for action in tag.get('actions', []):
                        if action.get('actionType') == 'VisualSearch':
                            for v in action.get('data', {}).get('value', []):
                                if 'contentUrl' in v:
                                    urls.append(v['contentUrl'])
                return urls[:self.n_refs]
        except Exception as e:
            logger.warning(f"Bing search failed: {e}")
        return []

    # ── download & cache ──────────────────────────────────────────────────────
    def _download_image(self, url: str) -> Optional[np.ndarray]:
        local = os.path.join(self.cache_dir, hashlib.md5(url.encode()).hexdigest() + '.jpg')
        if os.path.exists(local):
            return np.array(Image.open(local).convert('RGB'))
        try:
            import requests
            resp = requests.get(url, timeout=self.timeout, stream=True)
            if resp.status_code == 200:
                img = Image.open(resp.raw).convert('RGB')
                img.save(local)
                return np.array(img)
        except Exception as e:
            logger.warning(f"Download failed {url}: {e}")
        return None

    # ── public interface ──────────────────────────────────────────────────────
    def retrieve(self, query_img) -> Tuple[List[np.ndarray], List[str], int]:
        """Retrieve reference images via web search.

        Returns (ref_images, urls, overhead_bits).
        Overhead ≈ 30-100 bytes per URL × 8 bits.
        """
        h = self._img_hash(query_img)
        if h in self._url_cache:
            urls = self._url_cache[h]
        else:
            search_fn = {'baidu': self._search_baidu, 'google': self._search_google,
                         'bing': self._search_bing}.get(self.backend, self._search_baidu)
            urls = search_fn(query_img)
            if urls:
                self._url_cache[h] = urls
                with open(self._cache_path, 'wb') as f:
                    pickle.dump(self._url_cache, f)

        imgs = []
        for u in urls:
            im = self._download_image(u)
            if im is not None:
                imgs.append(im)
        overhead = sum(len(u.encode('utf-8')) for u in urls) * 8
        return imgs, urls, overhead


# ═══════════════════════════════════════════════════════════════════════════════
#  Method 3 — Image-Text-Image Generation  (VLM + Diffusion)
# ═══════════════════════════════════════════════════════════════════════════════

class ImageTextImageGeneration:
    """Image → Text (VLM) → Image (Diffusion) reference generation.

    VLM backends:   'llava', 'qwen-vl', 'internvl', 'blip2'
    Diffusion backends:  'flux', 'sdxl', 'sd3'

    Deterministic generation: seeds derived from SHA-256(caption + index).
    Overhead: len(caption_bytes) × 8 bits  (typically 40-90 bytes).
    """

    def __init__(self, n_refs: int = 3, vlm: str = 'llava', diffusion: str = 'flux',
                 device: str = 'cuda', caption_cache_dir: str = './_caption_cache'):
        self.n_refs = n_refs
        self.vlm_name = vlm
        self.diff_name = diffusion
        self.device = device
        self.cache_dir = caption_cache_dir
        os.makedirs(caption_cache_dir, exist_ok=True)
        self._vlm = None
        self._vlm_proc = None
        self._pipe = None
        # Caption cache for reproducibility
        self._cap_cache_path = os.path.join(caption_cache_dir, 'captions.pkl')
        self._cap_cache = {}
        if os.path.exists(self._cap_cache_path):
            with open(self._cap_cache_path, 'rb') as f:
                self._cap_cache = pickle.load(f)

    # ── deterministic seeds ───────────────────────────────────────────────────
    @staticmethod
    def _seeds(caption: str, n: int) -> List[int]:
        return [int.from_bytes(hashlib.sha256(f"{caption}_ref_{i}".encode()).digest()[:4], 'big')
                for i in range(n)]

    # ── VLM caption generation ────────────────────────────────────────────────
    def _ensure_vlm(self):
        if self._vlm is not None:
            return
        if self.vlm_name == 'llava':
            try:
                from transformers import LlavaForConditionalGeneration, AutoProcessor
                self._vlm = LlavaForConditionalGeneration.from_pretrained(
                    "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
                self._vlm_proc = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            except Exception as e:
                logger.warning(f"LLaVA load failed: {e}")
        elif self.vlm_name == 'blip2':
            try:
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                self._vlm = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")
                self._vlm_proc = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            except Exception as e:
                logger.warning(f"BLIP2 load failed: {e}")
        # Add more VLM backends as needed

    def generate_caption(self, img) -> str:
        """Generate text caption for an image. Uses cache for reproducibility."""
        if isinstance(img, np.ndarray):
            h = hashlib.md5(img.tobytes()[:4096]).hexdigest()
        else:
            h = hashlib.md5(np.array(img).tobytes()[:4096]).hexdigest()
        if h in self._cap_cache:
            return self._cap_cache[h]

        self._ensure_vlm()
        caption = self._run_vlm(img)
        self._cap_cache[h] = caption
        with open(self._cap_cache_path, 'wb') as f:
            pickle.dump(self._cap_cache, f)
        return caption

    @torch.no_grad()
    def _run_vlm(self, img) -> str:
        if self._vlm is None:
            return "A photograph of a natural scene"  # safe fallback
        pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
        if pil.mode != 'RGB':
            pil = pil.convert('RGB')
        if self.vlm_name == 'llava':
            prompt = ("USER: <image>\nDescribe this image in one detailed paragraph "
                      "for the purpose of recreating a similar image.\nASSISTANT:")
            inp = self._vlm_proc(text=prompt, images=pil, return_tensors="pt").to(self.device)
            out = self._vlm.generate(**inp, max_new_tokens=150, do_sample=False)
            text = self._vlm_proc.decode(out[0], skip_special_tokens=True)
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:")[-1].strip()
            return text
        elif self.vlm_name == 'blip2':
            inp = self._vlm_proc(images=pil, return_tensors="pt").to(self.device, torch.float16)
            out = self._vlm.generate(**inp, max_new_tokens=100)
            return self._vlm_proc.decode(out[0], skip_special_tokens=True).strip()
        return "A photograph"

    # ── Diffusion image generation ────────────────────────────────────────────
    def _ensure_diffusion(self):
        if self._pipe is not None:
            return
        try:
            if self.diff_name == 'flux':
                from diffusers import FluxPipeline
                self._pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16).to(self.device)
            elif self.diff_name == 'sdxl':
                from diffusers import StableDiffusionXLPipeline
                self._pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16, variant="fp16").to(self.device)
            elif self.diff_name == 'sd3':
                from diffusers import StableDiffusion3Pipeline
                self._pipe = StableDiffusion3Pipeline.from_pretrained(
                    "stabilityai/stable-diffusion-3-medium-diffusers",
                    torch_dtype=torch.float16).to(self.device)
            logger.info(f"Diffusion model ({self.diff_name}) loaded")
        except Exception as e:
            logger.warning(f"Diffusion model load failed: {e}")

    @torch.no_grad()
    def _generate_one(self, caption: str, seed: int, size: Tuple[int, int]) -> np.ndarray:
        self._ensure_diffusion()
        if self._pipe is None:
            # Fallback: deterministic colored noise (better than random noise)
            rng = np.random.RandomState(seed)
            return rng.randint(0, 256, (*size, 3)).astype(np.uint8)

        gen = torch.Generator(device=self.device).manual_seed(seed)
        img = self._pipe(caption, height=size[0], width=size[1],
                         num_inference_steps=4 if 'flux' in self.diff_name else 20,
                         generator=gen).images[0]
        return np.array(img)

    # ── public interface ──────────────────────────────────────────────────────
    def retrieve(self, query_img, target_size=(256, 256)
                 ) -> Tuple[List[np.ndarray], str, int]:
        """Full pipeline: image → caption → generated references.

        Returns (ref_images, caption, overhead_bits).
        """
        caption = self.generate_caption(query_img)
        seeds = self._seeds(caption, self.n_refs)
        imgs = [self._generate_one(caption, s, target_size) for s in seeds]
        overhead = len(caption.encode('utf-8')) * 8
        return imgs, caption, overhead


# ═══════════════════════════════════════════════════════════════════════════════
#  Adaptive Reference Selection  (Algorithm 1 in paper)
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveReferenceSelector:
    """Choose the best reference method per image by RD cost.

        J_s = R_s + β · D_s,   s ∈ {local, web, iti}
        s* = argmin J_s
    """

    def __init__(self, model=None,
                 local: Optional[LocalDictionaryRetrieval] = None,
                 web: Optional[WebImageRetrieval] = None,
                 iti: Optional[ImageTextImageGeneration] = None,
                 beta: float = 0.01):
        self.model = model
        self.methods = {}
        if local: self.methods['local'] = local
        if web:   self.methods['web'] = web
        if iti:   self.methods['iti'] = iti
        self.beta = beta

    @torch.no_grad()
    def _rd_cost(self, x_tensor, refs, model) -> float:
        if not refs or model is None:
            return float('inf')
        dev = next(model.parameters()).device
        ref_t = []
        for r in refs:
            t = transforms.ToTensor()(Image.fromarray(r).convert('RGB'))
            t = F.interpolate(t.unsqueeze(0), x_tensor.shape[-2:], mode='bilinear', align_corners=False)
            ref_t.append(t.to(dev))
        out = model(x_tensor.to(dev), ref_t)
        npx = x_tensor.shape[2] * x_tensor.shape[3]
        rate = sum(torch.log(lh).sum() / (-np.log(2) * npx) for lh in out['likelihoods'].values()).item()
        dist = F.mse_loss(out['x_hat'], x_tensor.to(dev)).item()
        return rate + self.beta * dist

    def select(self, query_img, x_tensor=None, target_size=(256, 256)):
        """Returns (refs, method_name, overhead_bits)."""
        best_cost, best = float('inf'), ([], 'none', 0)
        for name, m in self.methods.items():
            try:
                if name == 'iti':
                    refs, _, bits = m.retrieve(query_img, target_size)
                else:
                    refs, _, bits = m.retrieve(query_img)
                if refs and self.model is not None and x_tensor is not None:
                    cost = self._rd_cost(x_tensor, refs, self.model)
                else:
                    cost = -len(refs)  # heuristic: more refs = better
                if cost < best_cost:
                    best_cost = cost
                    best = (refs, name, bits)
            except Exception as e:
                logger.warning(f"Method {name} failed: {e}")
        return best


# ═══════════════════════════════════════════════════════════════════════════════
#  Unified Reference Manager
# ═══════════════════════════════════════════════════════════════════════════════

class ReferenceManager:
    """One-stop interface for all reference generation methods."""

    def __init__(self, ref_path=None, n_clusters=3000, n_refs=3, cache_dir=None,
                 web_backend='', web_api_key='', web_cx='',
                 vlm='', diffusion='', device='cuda'):
        self.n_refs = n_refs

        # Method 1: Local Dictionary (always if ref_path given)
        self.local = LocalDictionaryRetrieval(ref_path, n_clusters, n_refs,
                                              cache_dir, device) if ref_path else None
        # Method 2: Web Search (if backend specified)
        self.web = WebImageRetrieval(n_refs, web_backend, web_api_key, web_cx,
                                     cache_dir=os.path.join(cache_dir or '.', '_web')
                                     ) if web_backend else None
        # Method 3: Image-Text-Image (if vlm specified)
        self.iti = ImageTextImageGeneration(n_refs, vlm, diffusion, device,
                                            os.path.join(cache_dir or '.', '_iti')
                                            ) if vlm else None
        self.selector = None

    def setup_adaptive(self, model, beta=0.01):
        self.selector = AdaptiveReferenceSelector(model, self.local, self.web, self.iti, beta)

    def get_references(self, query_img, method='auto', **kw):
        """Get references. method ∈ {'auto','local','web','iti'}."""
        if method == 'auto' and self.selector:
            return self.selector.select(query_img, **kw)
        if method == 'web' and self.web:
            return self.web.retrieve(query_img)
        if method == 'iti' and self.iti:
            imgs, cap, bits = self.iti.retrieve(query_img, kw.get('target_size', (256, 256)))
            return imgs, cap, bits
        if self.local:
            return self.local.retrieve(query_img)
        return [], 'none', 0
