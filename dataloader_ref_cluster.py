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

# import pillow_heif

# pillow_heif.register_heif_opener()

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

        if os.path.isdir(path):
            self.keys = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

        elif path.endswith('.h5') or path.endswith('.hdf5'):
            with h5py.File(path, 'r') as f:
                self.keys = list(f.keys())
        else:
            raise ValueError("path must be either a directory or an HDF5 file")
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
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.webp', '.heic')):
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
            directory = os.path.dirname(self.feature_cache_path)
            os.makedirs(directory, exist_ok=True)
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
        # 获取样本 key
        key = self.keys[idx]
        sample = self.get_data(key)
        if len(sample.shape) == 2:
            sample = np.stack([sample] * 3, axis=-1)
        elif sample.shape[2] == 1:
            sample = np.concatenate([sample] * 3, axis=-1)
        
        # 提取样本特征
        sample_feature = self.extract_feature(sample)

        # 最近邻搜索匹配参考样本的 key
        _, indices = self.nn_searcher.kneighbors(sample_feature.reshape(1, -1))
        ref_keys = [self.feature_to_key[i] for i in indices[0]]

        ref_samples = []
        for ref_key in ref_keys:
            if isinstance(self.ref_data, dict):
                with Image.open(self.ref_data[ref_key]) as img:
                    ref_sample = np.array(img)
            else:
                ref_sample = self.ref_data[ref_key][:]
            ref_samples.append(ref_sample)

        # 将 sample 转换为 torch.Tensor 并归一化
        sample = self.normalize_to_tensor(sample)

        # 对 ref_samples 列表中的每个参考样本进行转换和归一化
        ref_samples = [self.normalize_to_tensor(ref_sample) for ref_sample in ref_samples]

        return sample, ref_samples, key, ref_keys

    def normalize_to_tensor(self, img):
        """
        将一个 numpy 数组 (uint8) 转换为 torch.Tensor 并归一化到 [0, 1] 范围。
        :param img: numpy array, uint8 格式的图像数组，值范围为 [0, 255]
        :return: torch.Tensor, float32 类型，值范围为 [0, 1]
        """
        # 如果 img 是 numpy 数组，确保它的类型为 uint8
        if isinstance(img, np.ndarray):
            tensor = torch.tensor(img, dtype=torch.float32) / 255.0
            tensor = tensor.permute(2, 0, 1)
            return tensor
        else:
            raise ValueError("Input image should be a numpy array.")

    def _get_data(self, key):
        if os.path.isdir(self.path):
            img_path = os.path.join(self.path, key)
            with Image.open(img_path) as img:
                img = np.array(img)
            return img
        elif self.path.endswith('.h5') or self.path.endswith('.hdf5'):
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
            # 如果 img 不是一个 Tensor，则将其转换为 PIL.Image
            if not isinstance(img, torch.Tensor):
                img = Image.fromarray(np.uint8(img))

                # 检查图像的模式，如果是灰度图（L 模式），则转换为 3 通道
                if img.mode == 'L':  # 'L' 表示灰度图
                    img = img.convert('RGB')  # 将灰度图转换成 3 通道的 RGB 图像

            # 应用 transform 进行数据预处理
            img = self.transform(img)

            # 将图像转换为 4D Tensor（用于批处理），并移动到指定设备
            img_tensor = img.unsqueeze(0).to(self.device)

            # 使用特征提取器提取特征
            feature = self.feature_extractor(img_tensor)

        # 返回特征，去除 batch 维度并转换为 CPU 上的 numpy 数组
        return feature.squeeze().cpu().numpy()

    def retrieve_similar_images(self, query_img_path, n_refs=None, save_visualization=None):
        """
        给定一张图片路径，检索最相似的图片
        
        Args:
            query_img_path: 查询图片路径
            n_refs: 返回的相似图片数量，如果为None则使用初始化时的n_refs
            save_visualization: 可视化结果保存路径，如果为None则不保存
            
        Returns:
            similar_images: 包含相似图片的列表
            similar_keys: 相似图片对应的键名列表
        """
        n_refs = n_refs or self.n_refs
        
        # 读取查询图片
        with Image.open(query_img_path) as img:
            query_img = np.array(img)
        
        # 提取特征
        query_feature = self.extract_feature(query_img)
        
        # 最近邻搜索
        _, indices = self.nn_searcher.kneighbors(query_feature.reshape(1, -1))
        similar_keys = [self.feature_to_key[i] for i in indices[0]]
        
        # 获取相似图片
        similar_images = []
        for key in similar_keys:
            if isinstance(self.ref_data, dict):
                with Image.open(self.ref_data[key]) as img:
                    similar_images.append(np.array(img))
            else:
                similar_images.append(self.ref_data[key][:])
        
        # 如果需要可视化结果
        if save_visualization:
            self._visualize_retrieval(query_img_path, similar_images, similar_keys, save_visualization)
        
        return similar_images, similar_keys
    
    def _visualize_retrieval(self, query_path, similar_images, similar_keys, save_path):
        """
        可视化检索结果
        
        Args:
            query_path: 查询图片路径
            similar_images: 相似图片列表
            similar_keys: 相似图片键名列表
            save_path: 保存路径
        """
        n_images = 1 + len(similar_images)
        fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
        
        # 显示查询图片
        query_img = Image.open(query_path)
        axes[0].imshow(query_img)
        axes[0].set_title('Query Image')
        axes[0].axis('off')
        
        # 显示相似图片
        for i, (img, key) in enumerate(zip(similar_images, similar_keys), 1):
            axes[i].imshow(img)
            axes[i].set_title(f'Similar Image {i}\n{key}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Visualization saved to {save_path}")


    # def __del__(self):
    #     # if hasattr(self.local, 'data'):
    #     #     self.local.data.close()
    #     if isinstance(self.ref_data, h5py.File):
    #         self.ref_data.close()

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

def test_dataset_for_missing_values(dataset):
    """
    测试数据集中是否存在缺失值
    """
    print("开始测试数据集...")
    
    # 测试样本数量
    test_size = min(100, len(dataset))  # 测试前100个样本或整个数据集
    # test_size = len(dataset)
    errors = []
    
    # 1. 测试keys是否存在
    print(f"\n1. 测试数据集keys...")
    if not dataset.keys:
        errors.append("数据集keys列表为空")
    else:
        print(f"数据集包含 {len(dataset.keys)} 个样本")
    
    # 2. 测试参考数据
    print("\n2. 测试参考数据...")
    if isinstance(dataset.ref_data, dict):
        missing_refs = [k for k in dataset.ref_keys if k not in dataset.ref_data]
        if missing_refs:
            errors.append(f"参考数据中缺失以下keys: {missing_refs}")
    
    # 3. 测试特征数据
    print("\n3. 测试特征数据...")
    if dataset.ref_features is None:
        errors.append("特征数据为空")
    else:
        # 检查特征向量是否包含NaN值
        if np.any(np.isnan(dataset.ref_features)):
            errors.append("特征向量中存在NaN值")
        # 检查特征向量是否包含无穷大值
        if np.any(np.isinf(dataset.ref_features)):
            errors.append("特征向量中存在无穷大值")
    
    # 4. 测试数据加载
    print("\n4. 测试数据加载...")
    for idx in tqdm(range(test_size), desc="测试样本加载"):
        try:
            # 测试数据加载
            sample, ref_samples, key, ref_keys = dataset[idx]
            
            # 检查样本数据
            if sample is None:
                errors.append(f"样本 {idx} (key: {key}) 为空")
            elif torch.isnan(sample).any():
                errors.append(f"样本 {idx} (key: {key}) 包含NaN值")
                
            # 检查参考样本
            for i, ref_sample in enumerate(ref_samples):
                if ref_sample is None:
                    errors.append(f"样本 {idx} 的参考样本 {i} 为空")
                elif torch.isnan(ref_sample).any():
                    errors.append(f"样本 {idx} 的参考样本 {i} 包含NaN值")
                    
            # 检查图像尺寸是否符合patch_size要求
            if sample.shape[-2:] != (256, 256):
                errors.append(f"样本 {idx} 尺寸不符合要求: {sample.shape}")
            
        except Exception as e:
            errors.append(f"处理样本 {idx} 时发生错误: {str(e)}")
    
    # 5. 输出测试结果
    print("\n=== 测试结果 ===")
    if errors:
        print("发现以下问题：")
        for error in errors:
            print(f"- {error}")
    else:
        print("未发现任何缺失值或异常")
    
    return len(errors) == 0




# 使用示例
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试数据集是否存在缺失值.')
    parser.add_argument('--data_path', type=str, 
                       default='/h3cstore_ns/ydchen/DATASET/kodak.hdf5', 
                       help='主数据集路径')
    parser.add_argument('--ref_path', type=str, 
                       default='/h3cstore_ns/ydchen/DATASET/coding_img_cropped_2/Flickr2K.hdf5', 
                       help='参考数据集路径')
    parser.add_argument('--feature_cache_path', type=str, 
                       default='/h3cstore_ns/ydchen/code/CompressAI/LIC_TCM/model_ckpt_TCM/data_cluster_feature/flicker_features.pkl', 
                       help='特征缓存路径')
    parser.add_argument('--n_clusters', type=int, default=3000, help='聚类数量')
    parser.add_argument('--n_refs', type=int, default=3, help='参考图像数量')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256], help='图片块大小')
    parser.add_argument('--cuda', action='store_true', help='是否使用CUDA')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载的工作进程数')
    parser.add_argument('--query_image', type=str, default='/h3cstore_ns/ydchen/DATASET/kodak/kodim05.png', help='用于测试图片检索的查询图片路径')

    args = parser.parse_args()
    
    # 设置设备
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    
    # 初始化数据集
    transform = transforms.Compose([
        transforms.Resize(args.patch_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LICDataset(
        path=args.data_path,
        ref_path=args.ref_path,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        feature_cache_path=args.feature_cache_path,
        n_clusters=args.n_clusters,
        n_refs=args.n_refs
    )
    
    # 运行测试
    # test_result = test_dataset_for_missing_values(dataset)
    # print(f"\n测试{'通过' if test_result else '失败'}")
    if args.query_image:
        print("\nTesting image retrieval:")
        similar_images, similar_keys = dataset.retrieve_similar_images(
            args.query_image,
            n_refs=args.n_refs,
            save_visualization=os.path.join('/h3cstore_ns/ydchen/code/CompressAI/LIC_TCM/CLC_trained_model_1224', 'retrieval_result.png')
        )
        print(f"Found {len(similar_images)} similar images")
        print(f"Similar image keys: {similar_keys}")