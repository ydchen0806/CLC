import h5py 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import pickle
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import random

class LICDataset(torch.utils.data.Dataset):
    def __init__(self, path, ref_path, transform=None, device='cuda', batch_size=1000, cache_size=1024, feature_cache_path=None, n_clusters=None, n_refs=1, use_spp=True):
        self.path = path # 图像的路径
        self.ref_path = ref_path # 参考的路径
        self.device = device
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.n_refs = n_refs
        self.use_spp = use_spp
        
        # 确保 feature_cache_path 是一个目录
        if feature_cache_path:
            if os.path.isfile(feature_cache_path):
                feature_cache_path = os.path.dirname(feature_cache_path)
            self.feature_cache_dir = feature_cache_path
        else:
            self.feature_cache_dir = None

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.feature_extractor = models.resnet50(pretrained=True)
        if self.use_spp != 1:
            self.feature_extractor.fc = nn.Identity() # 输出倒数第二层的特征
        self.feature_extractor = self.feature_extractor.to(device)
        self.feature_extractor.eval()

        with h5py.File(path, 'r') as f:
            self.keys = list(f.keys())
        self.len = len(self.keys) # dataset len

        self.ref_data, self.ref_keys = self.load_ref_data(ref_path) # ref_data: dict, ref_keys: list

        self.pca = None  # 初始化为 None
        self.ref_features, self.feature_to_key = self.load_or_compute_features()
        
        if self.n_clusters and self.n_clusters > 100:
            self.cluster_features()
        print(f"Using {len(self.ref_features)} representative features.")
        print(f"Feature shape: {self.ref_features.shape}, Key shape: {len(self.feature_to_key)}")
        
        self.nn_searcher = NearestNeighbors(n_neighbors=self.n_refs, algorithm='ball_tree', n_jobs=-1)
        self.nn_searcher.fit(self.ref_features)

        self.local = threading.local()
        
        self.get_data = lru_cache(maxsize=cache_size)(self._get_data)

    def load_ref_data(self, ref_path):
        if os.path.isdir(ref_path):
            print("Loading reference data from directory...")
            ref_data = {}
            ref_keys = []
            for filename in os.listdir(ref_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    img_path = os.path.join(ref_path, filename)
                    ref_data[filename] = img_path
                    ref_keys.append(filename)
            return ref_data, ref_keys
        elif ref_path.endswith('.h5') or ref_path.endswith('.hdf5'):
            print("Loading reference data from HDF5 file...")
            ref_data = h5py.File(ref_path, 'r')
            ref_keys = list(ref_data.keys())
            return ref_data, ref_keys
        else:
            raise ValueError("ref_path must be either a directory or an HDF5 file")

    def load_or_compute_features(self):
        feature_type = "spp" if self.use_spp == 1 else "resnet"
        pca_info = "_pca256"  # 假设PCA组件数为256
        feature_cache_path = os.path.join(self.feature_cache_dir, f'features_{feature_type}{pca_info}.pkl') if self.feature_cache_dir else None
        
        if feature_cache_path and os.path.exists(feature_cache_path):
            print(f"Loading pre-computed features from {feature_cache_path}...")
            with open(feature_cache_path, 'rb') as f:
                data = pickle.load(f)
                features, feature_to_key, self.pca = data
        else:
            print("Computing features...")
            features, feature_to_key = self.precompute_features() # (1*2048*64) (2048*64)
            
            # 确保特征是二维的
            features = features.reshape(features.shape[0], -1)
            
            # 使用PCA降维
            print("Applying PCA...")
            self.pca = PCA(n_components=256)
            reduced_features = self.pca.fit_transform(features) # 特征降维
            
            if feature_cache_path:
                print(f"Saving computed features to {feature_cache_path}...")
                os.makedirs(os.path.dirname(feature_cache_path), exist_ok=True)
                with open(feature_cache_path, 'wb') as f:
                    pickle.dump((reduced_features, feature_to_key, self.pca), f)
            
            features = reduced_features
        
        return features, feature_to_key

    def cluster_features(self):
        feature_type = "spp" if self.use_spp == 1 else "resnet"
        pca_info = "_pca256"  # 假设PCA组件数为256
        cluster_cache_path = os.path.join(self.feature_cache_dir, f'cluster_cache_{self.n_clusters}_{feature_type}{pca_info}.pkl') if self.feature_cache_dir else None
        
        if cluster_cache_path and os.path.exists(cluster_cache_path):
            print(f"Loading cached clustering results from {cluster_cache_path}")
            with open(cluster_cache_path, 'rb') as f:
                cache = pickle.load(f)
                self.ref_features = cache['ref_features']
                self.feature_to_key = cache['feature_to_key']
        else:
            print(f"Clustering features into {self.n_clusters} clusters...")
            kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=1000)
            cluster_labels = kmeans.fit_predict(self.ref_features)
            
            new_features = []
            new_feature_to_key = {}
            
            for i in tqdm(range(self.n_clusters), desc="Processing clusters"):
                cluster_indices = np.where(cluster_labels == i)[0]
                cluster_center = kmeans.cluster_centers_[i]
                
                # Find the feature closest to the cluster center
                distances = np.linalg.norm(self.ref_features[cluster_indices] - cluster_center, axis=1)
                closest_index = cluster_indices[np.argmin(distances)]
                
                new_features.append(self.ref_features[closest_index])
                new_feature_to_key[len(new_features) - 1] = self.feature_to_key[closest_index]
            
            self.ref_features = np.array(new_features)
            self.feature_to_key = new_feature_to_key
            
            if cluster_cache_path:
                print(f"Saving clustering results to {cluster_cache_path}...")
                os.makedirs(os.path.dirname(cluster_cache_path), exist_ok=True)
                with open(cluster_cache_path, 'wb') as f:
                    pickle.dump({
                        'ref_features': self.ref_features,
                        'feature_to_key': self.feature_to_key
                    }, f)
        
        print(f"Clustering complete. Now using {len(self.ref_features)} representative features.")

    def __len__(self):
        return self.len # 10000
    
    def __getitem__(self, idx):  #[0,9999]
        key = self.keys[idx] # X_id
        sample = self.get_data(key) # X
        
        # 多查询策略
        sample_feature = self.extract_feature(sample)
        rotated_sample = TF.rotate(Image.fromarray(sample), 90)
        rotated_feature = self.extract_feature(rotated_sample)
        
        _, indices1 = self.nn_searcher.kneighbors(sample_feature.reshape(1, -1))
        _, indices2 = self.nn_searcher.kneighbors(rotated_feature.reshape(1, -1))
        
        indices = np.unique(np.concatenate([indices1[0], indices2[0]]))[:self.n_refs]
        ref_keys = [self.feature_to_key[i] for i in indices]
        
        ref_samples = []
        for ref_key in ref_keys:
            if isinstance(self.ref_data, dict):
                with Image.open(self.ref_data[ref_key]) as img:
                    ref_sample = np.array(img)
            else:
                ref_sample = self.ref_data[ref_key][()]
            ref_samples.append(ref_sample)
        
        return sample, ref_samples, key, ref_keys

    def _get_data(self, key):
        if not hasattr(self.local, 'data'):
            self.local.data = h5py.File(self.path, 'r')
        return self.local.data[key][()]

    def precompute_features(self):
        features = []
        feature_to_key = {}
        
        def process_batch(batch): # 1000
            batch_features = []
            batch_keys = []
            for key in batch:
                if isinstance(self.ref_data, dict):
                    with Image.open(self.ref_data[key]) as img:
                        img = np.array(img)
                else:
                    img = self.ref_data[key][()]
                feature = self.extract_feature(img)
                batch_features.append(feature)
                batch_keys.append(key)
            return batch_features, batch_keys # dict

        total_batches = (len(self.ref_keys) + self.batch_size - 1) // self.batch_size
        
        with ThreadPoolExecutor() as executor: # 多线程提速
            futures = []
            for i in range(0, len(self.ref_keys), self.batch_size):
                batch = self.ref_keys[i:i+self.batch_size]
                futures.append(executor.submit(process_batch, batch))
            
            for i, future in tqdm(enumerate(futures), total=total_batches, desc="Computing features"):
                batch_features, batch_keys = future.result()
                features.extend(batch_features)
                for j, key in enumerate(batch_keys):
                    feature_to_key[i * self.batch_size + j] = key
        
        return np.array(features), feature_to_key

    def spatial_pyramid_pooling(self, x, levels=[1, 2, 4]):
        features = []
        for level in levels:
            h = F.adaptive_max_pool2d(x, output_size=(level, level))
            h = h.view(h.size(0), -1)
            features.append(h)
        return torch.cat(features, dim=1)

    def extract_feature(self, img):
        with torch.no_grad():
            if not isinstance(img, torch.Tensor):
                img = Image.fromarray(np.uint8(img))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = self.transform(img)
            img_tensor = img.unsqueeze(0).to(self.device)
            
            x = self.feature_extractor.conv1(img_tensor)
            x = self.feature_extractor.bn1(x)
            x = self.feature_extractor.relu(x)
            x = self.feature_extractor.maxpool(x)

            x = self.feature_extractor.layer1(x)
            x = self.feature_extractor.layer2(x)
            x = self.feature_extractor.layer3(x)
            x = self.feature_extractor.layer4(x)
            
            if self.use_spp == 1:
                feature = self.spatial_pyramid_pooling(x)
            else:
                feature = F.adaptive_avg_pool2d(x, (1, 1))
                feature = feature.view(feature.size(0), -1)
            
            feature = feature.cpu().numpy().flatten()
            
            # 如果PCA已初始化，则应用PCA转换
            if self.pca is not None:
                feature = self.pca.transform(feature.reshape(1, -1)).flatten()
            
        return feature
    def process_and_save_single_image(self, image_path, output_dir):
        """
        Process a single input image, retrieve similar reference images, and save all images.
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载并预处理输入图像
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            input_img = np.array(img)
        
        # 保存输入图像
        input_name = os.path.basename(image_path)
        input_save_path = os.path.join(output_dir, f"input_{input_name}")
        Image.fromarray(input_img).save(input_save_path)
        
        # 提取特征并找到最近邻
        input_feature = self.extract_feature(input_img)
        _, indices = self.nn_searcher.kneighbors(input_feature.reshape(1, -1))
        indices = indices[0][:self.n_refs]
        ref_keys = [self.feature_to_key[i] for i in indices]
        
        # 加载并保存参考图像
        ref_samples = []
        ref_paths = []
        for i, ref_key in enumerate(ref_keys):
            if isinstance(self.ref_data, dict):
                with Image.open(self.ref_data[ref_key]) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    ref_sample = np.array(img)
            else:
                ref_sample = self.ref_data[ref_key][()]
                if ref_sample.shape[-1] == 4:  # If RGBA
                    ref_sample = ref_sample[..., :3]  # Keep only RGB channels
            
            ref_samples.append(ref_sample)
            ref_save_path = os.path.join(output_dir, f"ref_{i}_{ref_key}.png")
            Image.fromarray(ref_sample).save(ref_save_path)
            ref_paths.append(ref_save_path)
        
        return input_img, ref_samples, input_name, ref_keys, input_save_path, ref_paths

    def visualize_single_image_comparison(self, image_path, output_dir):
        """
        Visualize the comparison between a single input image and its retrieved reference images.
        Save individual images and the combined visualization.
        """
        input_img, ref_samples, input_name, ref_keys, input_path, ref_paths = self.process_and_save_single_image(image_path, output_dir)
        
        # 创建合并的可视化
        fig, axes = plt.subplots(1, 1 + len(ref_samples), figsize=(5 * (1 + len(ref_samples)), 5))
        
        axes[0].imshow(input_img)
        axes[0].set_title(f'Input Image\n{input_name}')
        axes[0].axis('off')
        
        for i, (ref_sample, ref_key) in enumerate(zip(ref_samples, ref_keys), 1):
            axes[i].imshow(ref_sample)
            axes[i].set_title(f'Reference Image {i}\n{ref_key}')
            axes[i].axis('off')
        
        plt.tight_layout()
        combined_save_path = os.path.join(output_dir, 'combined_comparison.png')
        plt.savefig(combined_save_path)
        plt.close()
        
        print(f"Input image saved to: {input_path}")
        print(f"Reference images saved to: {', '.join(ref_paths)}")
        print(f"Combined comparison saved to: {combined_save_path}")

    def visualize_comparison(self, idx, save_path):
        sample, ref_samples, sample_key, ref_keys = self[idx]
        
        fig, axes = plt.subplots(1, 1 + len(ref_samples), figsize=(5 * (1 + len(ref_samples)), 5))
        
        axes[0].imshow(sample)
        axes[0].set_title(f'Original Image\n{sample_key}')
        axes[0].axis('off')
        
        for i, (ref_sample, ref_key) in enumerate(zip(ref_samples, ref_keys), 1):
            axes[i].imshow(ref_sample)
            axes[i].set_title(f'Reference Image {i}\n{ref_key}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Comparison saved to {save_path}")

    def process_single_image(self, image_path):
        """
        Process a single input image and retrieve similar reference images.
        """
        # Load and preprocess the input image
        with Image.open(image_path) as img:
            # Convert image to RGB if it's not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            input_img = np.array(img)
        
        # Extract feature from the input image
        input_feature = self.extract_feature(input_img)
        
        # Find nearest neighbors
        _, indices = self.nn_searcher.kneighbors(input_feature.reshape(1, -1))
        indices = indices[0][:self.n_refs]
        ref_keys = [self.feature_to_key[i] for i in indices]
        
        # Load reference images
        ref_samples = []
        for ref_key in ref_keys:
            if isinstance(self.ref_data, dict):
                with Image.open(self.ref_data[ref_key]) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    ref_sample = np.array(img)
            else:
                ref_sample = self.ref_data[ref_key][()]
                if ref_sample.shape[-1] == 4:  # If RGBA
                    ref_sample = ref_sample[..., :3]  # Keep only RGB channels
            ref_samples.append(ref_sample)
        
        return input_img, ref_samples, os.path.basename(image_path), ref_keys


    def batch_visualize(self, indices, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for idx in indices:
            save_path = os.path.join(save_dir, f'comparison_{idx}.png')
            self.visualize_comparison(idx, save_path)

    def __del__(self):
        if hasattr(self, 'local') and hasattr(self.local, 'data'):
            self.local.data.close()
        if hasattr(self, 'ref_data') and isinstance(self.ref_data, h5py.File):
            self.ref_data.close()

def get_output_dir(base_dir, n_clusters, n_refs, use_spp):
    feature_type = "spp" if use_spp == 1 else "resnet"
    return os.path.join(base_dir, f"clusters_{n_clusters}_refs_{n_refs}_{feature_type}")

def main(args):
    # 设置路径
    data_path = args.data_path
    ref_path = args.ref_path
    feature_cache_path = args.feature_cache_path

    # 确保 feature_cache_path 是一个目录
    if feature_cache_path and os.path.isfile(feature_cache_path):
        feature_cache_path = os.path.dirname(feature_cache_path)
    
    # 根据 n_clusters, n_refs 和 use_spp 生成输出目录
    output_dir = get_output_dir(args.output_base_dir, args.n_clusters, args.n_refs, args.use_spp)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Feature cache directory: {feature_cache_path}")

    # 初始化数据集
    print("Initializing dataset...")
    dataset = LICDataset(data_path, ref_path, 
                         feature_cache_path=feature_cache_path,
                         n_clusters=args.n_clusters,
                         n_refs=args.n_refs,
                         use_spp=args.use_spp)
    
    print(f"Dataset initialized with {len(dataset)} samples.")
    print(f"Using {'SPP' if args.use_spp == 1 else 'ResNet'} features")

    if args.input_image:
        print(f"\nProcessing single input image: {args.input_image}")
        single_image_output_dir = os.path.join(output_dir, 'single_image_results')
        dataset.visualize_single_image_comparison(args.input_image, single_image_output_dir)
        print(f"Single image results saved in: {single_image_output_dir}")
    else:
        # 可视化一些比较结果
        print("Visualizing comparisons...")
        num_comparisons = args.num_comparisons
        total_samples = len(dataset)
        # set random seed
        random.seed(42)
        # 随机选择 num_comparisons 个不重复的索引
        random_indices = random.sample(range(total_samples), num_comparisons)

        print(f"Randomly selected indices for visualization: {random_indices}")
        dataset.batch_visualize(random_indices, output_dir)

        print(f"Comparisons saved in {output_dir}")

        # 测试获取单个样本
        print("\nTesting single sample retrieval:")
        sample, ref_samples, sample_key, ref_keys = dataset[0]
        print(f"Sample key: {sample_key}")
        print(f"Reference keys: {ref_keys}")
        print(f"Sample shape: {sample.shape}")
        print(f"Reference sample shapes: {[ref.shape for ref in ref_samples]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset and visualize comparisons.')
    parser.add_argument('--data_path', type=str, default='/img_video/img/Flicker2W.hdf5', help='Path to the main dataset')
    parser.add_argument('--ref_path', type=str, default='/img_video/img/Flicker2K.hdf5', help='Path to the reference dataset')
    parser.add_argument('--feature_cache_path', type=str, default='/h3cstore_ns/ydchen/code/CompressAI/data_cluster_feature_0715', help='Directory for feature cache')
    parser.add_argument('--output_base_dir', type=str, default='/h3cstore_ns/ydchen/code/CompressAI/data_cluster_feature_0813/comparison_results', help='Base output directory for results')
    parser.add_argument('--n_clusters', type=int, default=0, help='Number of clusters')
    parser.add_argument('--n_refs', type=int, default=10, help='Number of reference images')
    parser.add_argument('--num_comparisons', type=int, default=10, help='Number of comparisons to visualize')
    parser.add_argument('--use_spp', type=int, default=1, help='Use spatial pyramid pooling')
    parser.add_argument('--input_image', type=str, help='Path to a single input image for processing', default='/h3cstore_ns/ydchen/code/CompressAI/联想截图_20240812165416.png')

    args = parser.parse_args()
    main(args)