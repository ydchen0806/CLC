import h5py 
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
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
    def __init__(self, path, ref_path, transform=None, device='cuda', batch_size=1000, cache_size=1024, feature_cache_path=None, n_clusters=None, n_refs=1):
        self.path = path
        self.ref_path = ref_path
        self.device = device
        self.batch_size = batch_size
        self.feature_cache_path = feature_cache_path
        self.n_clusters = n_clusters
        self.n_refs = n_refs
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        self.feature_extractor = self.feature_extractor.to(device)
        self.feature_extractor.eval()

        with h5py.File(path, 'r') as f:
            self.keys = list(f.keys())
        self.len = len(self.keys)

        self.ref_data, self.ref_keys = self.load_ref_data(ref_path)

        self.ref_features, self.feature_to_key = self.load_or_compute_features()
        
        if self.n_clusters:
            self.cluster_features()
        print(f"Using {len(self.ref_features)} representative features.")
        print(f"Feature shape: {self.ref_features.shape}, Key shape: {len(self.feature_to_key)}, start clustering...")
        self.nn_searcher = NearestNeighbors(n_neighbors=self.n_refs, algorithm='ball_tree').fit(self.ref_features)

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
        if self.feature_cache_path and os.path.exists(self.feature_cache_path):
            print("Loading pre-computed features...")
            with open(self.feature_cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            print("Computing features...")
            features, feature_to_key = self.precompute_features()
            if self.feature_cache_path:
                print("Saving computed features...")
                with open(self.feature_cache_path, 'wb') as f:
                    pickle.dump((features, feature_to_key), f)
            return features, feature_to_key

    def cluster_features(self):
        basename = os.path.dirname(self.feature_cache_path)
        cluster_cache_path = f'{basename}/cluster_cache_{self.n_clusters}.pkl'
        
        if os.path.exists(cluster_cache_path):
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
            
            print("Saving clustering results to cache...")
            with open(cluster_cache_path, 'wb') as f:
                pickle.dump({
                    'ref_features': self.ref_features,
                    'feature_to_key': self.feature_to_key
                }, f)
        
        print(f"Clustering complete. Now using {len(self.ref_features)} representative features.")

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        sample = self.get_data(key)
        
        sample_feature = self.extract_feature(sample)
        
        _, indices = self.nn_searcher.kneighbors(sample_feature.reshape(1, -1))
        ref_keys = [self.feature_to_key[i] for i in indices[0]]
        
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
        
        def process_batch(batch):
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
            return batch_features, batch_keys

        total_batches = (len(self.ref_keys) + self.batch_size - 1) // self.batch_size
        
        with ThreadPoolExecutor() as executor:
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

    def extract_feature(self, img):
        with torch.no_grad():
            if not isinstance(img, torch.Tensor):
                img = Image.fromarray(np.uint8(img))
                img = self.transform(img)
            img_tensor = img.unsqueeze(0).to(self.device)
            feature = self.feature_extractor(img_tensor)
        return feature.squeeze().cpu().numpy()

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

    def batch_visualize(self, indices, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for idx in indices:
            save_path = os.path.join(save_dir, f'comparison_{idx}.png')
            self.visualize_comparison(idx, save_path)

    def __del__(self):
        if hasattr(self.local, 'data'):
            self.local.data.close()
        if isinstance(self.ref_data, h5py.File):
            self.ref_data.close()

def get_output_dir(base_dir, n_clusters, n_refs):
    return os.path.join(base_dir, f"clusters_{n_clusters}_refs_{n_refs}")

def main(args):
    # 设置路径
    data_path = args.data_path
    ref_path = args.ref_path
    feature_cache_path = args.feature_cache_path
    
    # 根据 n_clusters 和 n_refs 生成输出目录
    output_dir = get_output_dir(args.output_base_dir, args.n_clusters, args.n_refs)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # 初始化数据集
    print("Initializing dataset...")
    dataset = LICDataset(data_path, ref_path, 
                         feature_cache_path=feature_cache_path,
                         n_clusters=args.n_clusters,
                         n_refs=args.n_refs)
    
    print(f"Dataset initialized with {len(dataset)} samples.")

    # 可视化一些比较结果
    print("Visualizing comparisons...")
    num_comparisons = args.num_comparisons
    total_samples = len(dataset)

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
    parser.add_argument('--feature_cache_path', type=str, default='/h3cstore_ns/ydchen/code/CompressAI/data_cluster_feature/flicker_features.pkl', help='Path to feature cache')
    parser.add_argument('--output_base_dir', type=str, default='/h3cstore_ns/ydchen/code/CompressAI/data_cluster_feature/comparison_results', help='Base output directory for results')
    parser.add_argument('--n_clusters', type=int, default=1000, help='Number of clusters')
    parser.add_argument('--n_refs', type=int, default=3, help='Number of reference images')
    parser.add_argument('--num_comparisons', type=int, default=10, help='Number of comparisons to visualize')

    args = parser.parse_args()
    main(args)