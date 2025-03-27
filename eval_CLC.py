import torch
import torch.nn.functional as F
from torchvision import transforms
from models import TCM, CLC
from dataloader_ref_cluster import LICDataset
import warnings
import os
import sys
import math
import argparse
from torch.utils.data import Dataset, DataLoader
import time
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
from PIL import Image
import csv
from datetime import datetime
from torchvision.transforms.functional import resize
warnings.filterwarnings("ignore")
import random


class KodakDataset(Dataset):
    def __init__(self, kodak_path, ref_base_path, crop_size=512, train_mode=True):
        """
        Initialize Kodak dataset
        
        Args:
            kodak_path: Path to Kodak dataset
            ref_base_path: Path to reference images "/h3cstore_ns/ydchen/code/CompressAI/LIC_TCM/step-3000_kodak"
            crop_size: Crop size, default is 512
            train_mode: Whether in training mode, default is True
        """
        self.kodak_path = kodak_path
        self.ref_base_path = ref_base_path
        self.crop_size = crop_size
        self.train_mode = train_mode
        
        # Get all images in Kodak dataset
        self.image_files = [f for f in os.listdir(kodak_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()  # Ensure consistent order
        
        # Transformations
        self.resize_transform = transforms.Resize(crop_size)
        self.center_crop = transforms.CenterCrop(crop_size)
        self.to_tensor = transforms.ToTensor()
        
        print(f"Loaded {len(self.image_files)} Kodak images")
    
    def __len__(self):
        if not self.train_mode:
            return len(self.image_files)
        else:
            return 48
    
    def random_crop(self, img, crop_size):
        """Random crop for the image"""
        # If image size is smaller than crop size, resize first
        if img.width < crop_size or img.height < crop_size:
            resize_size = max(crop_size, img.width, img.height)
            img = transforms.Resize(resize_size)(img)
        
        # Calculate maximum x and y coordinates for cropping
        max_x = img.width - crop_size
        max_y = img.height - crop_size
        
        # Randomly select top-left corner
        x = random.randint(0, max(0, max_x))
        y = random.randint(0, max(0, max_y))
        
        # Crop the image
        return img.crop((x, y, x + crop_size, y + crop_size))
    
    def process_image(self, img):
        """Process image based on mode"""
        if self.train_mode:
            # Training mode: random crop
            processed_img = self.random_crop(img, self.crop_size)
        else:
            # Inference mode: if image is smaller than crop size, resize first
            if img.width < self.crop_size or img.height < self.crop_size:
                resize_size = max(self.crop_size, img.width, img.height)
                img = self.resize_transform(img)
                processed_img = self.center_crop(img)
            else:
                processed_img = img  # No cropping in inference mode
        
        # Convert to tensor
        return self.to_tensor(processed_img)
    
    def __getitem__(self, idx):
        # Get original image
        total_num = len(self.image_files)
        if self.train_mode:
            idx = idx % total_num
        img_name = self.image_files[idx]
        img_path = os.path.join(self.kodak_path, img_name)
        
        # Extract image number (assuming format "kodimXX.png")
        img_num = int(''.join(filter(str.isdigit, img_name)))
        
        # Read original image
        with Image.open(img_path) as img:
            image = self.process_image(img)
        
        # Get 3 reference images
        ref_images = []
        for ref_idx in range(0, 3):  # Get 3 reference images
            ref_filename = f"{img_num-1}-{ref_idx}.png"
            ref_path = os.path.join(self.ref_base_path, ref_filename)
            
            if os.path.exists(ref_path):
                with Image.open(ref_path) as ref_img:
                    ref_tensor = self.process_image(ref_img)
                    ref_images.append(ref_tensor)
            else:
                # If reference image doesn't exist, create a zero tensor as replacement
                if self.train_mode:
                    # Fixed size in training mode
                    ref_tensor = torch.zeros((3, self.crop_size, self.crop_size))
                else:
                    # Same size as original in inference mode
                    ref_tensor = torch.zeros_like(image)
                ref_images.append(ref_tensor)
                print(f"Warning: Reference image {ref_path} doesn't exist")
        
        return image, ref_images, img_name


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def extract_bitrate_from_checkpoint(checkpoint_path):
    """Extract bitrate value from checkpoint path"""
    # Use regex to match numbers (including decimals)
    match = re.search(r'([0-9]+\.?[0-9]*?)checkpoint', os.path.basename(checkpoint_path))
    if match:
        return match.group(1)
    return None

def get_results_dir(base_dir):
    """Create a results directory for RD curve"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(base_dir, f'rd_curve_results_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def get_all_checkpoints(base_dir="/h3cstore_ns/ydchen/code/CompressAI/LIC_TCM/CLC_models"):
    """Get all best checkpoints from all bitrate directories"""
    # Find all subdirectories with pattern 0322_*
    checkpoint_dirs = glob.glob(os.path.join(base_dir, "0322_*"))
    
    checkpoints = []
    for dir_path in checkpoint_dirs:
        # Extract bitrate from directory name
        dir_name = os.path.basename(dir_path)
        bitrate = dir_name.replace("0322_", "")
        
        # Find the corresponding best checkpoint
        checkpoint_path = os.path.join(dir_path, f"{bitrate}checkpoint_best.pth.tar")
        if os.path.exists(checkpoint_path):
            checkpoints.append({
                'path': checkpoint_path,
                'bitrate': float(bitrate) if bitrate.replace('.', '', 1).isdigit() else 0
            })
    
    # Sort checkpoints by bitrate
    checkpoints.sort(key=lambda x: x['bitrate'])
    return checkpoints

def plot_rd_curve(results, save_path):
    """Plot rate-distortion curve for the evaluated checkpoints"""
    # Convert results to numpy arrays for plotting
    bitrates = np.array([r['bitrate'] for r in results])
    psnrs = np.array([r['psnr'] for r in results])
    
    # Sort by bitrate for correct plotting
    sort_idx = np.argsort(bitrates)
    bitrates = bitrates[sort_idx]
    psnrs = psnrs[sort_idx]
    
    # Plot the RD curve
    plt.figure(figsize=(10, 6))
    plt.plot(bitrates, psnrs, 'o-', linewidth=2, markersize=8)
    
    # Add annotations for each point
    for i, (br, ps) in enumerate(zip(bitrates, psnrs)):
        plt.annotate(f"{br:.3f}", 
                    (br, ps), 
                    textcoords="offset points",
                    xytext=(0, 10), 
                    ha='center')
    
    # Set labels and title
    plt.xlabel('Bitrate (bpp)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Rate-Distortion Performance', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Reverse x-axis (from low to high bitrate)
    plt.xscale('log')
    plt.gca().invert_xaxis()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"RD curve saved to {save_path}")

def resize_reference_images(ref_samples, target_size):
    """Resize all reference images to match the target size"""
    resized_refs = []
    for ref in ref_samples:
        # Get target dimensions
        _, _, target_h, target_w = target_size
        
        # Resize the reference image to match target dimensions
        resized_ref = resize(ref, [target_h, target_w])
        resized_refs.append(resized_ref)
    
    return resized_refs

def evaluate_checkpoint(checkpoint_path, data_path, ref_path, feature_cache_path, device, n_clusters=3000, n_refs=3):
    """Evaluate a single checkpoint and return the results"""
    p = 128  # Padding size
    
    # Initialize model
    net = CLC(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)
    
    # Load dataset
    # test_dataset = LICDataset(
    #     data_path,
    #     ref_path,
    #     transform=transforms.ToTensor(),
    #     feature_cache_path=feature_cache_path,
    #     n_clusters=n_clusters,
    #     n_refs=n_refs
    # )
    test_dataset = KodakDataset(
        kodak_path="/h3cstore_ns/ydchen/DATASET/kodak",
        ref_base_path="/h3cstore_ns/ydchen/code/CompressAI/LIC_TCM/step-3000_kodak",
        crop_size=512,  # Larger size for testing
        train_mode=False
    )
    
    net = net.to(device)
    net.eval()
    
    # Load checkpoint
    print(f"Loading {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dictory = {}
    for k, v in checkpoint["state_dict"].items():
        dictory[k.replace("module.", "")] = v
    net.load_state_dict(dictory)
    net.update()
    
    # Track metrics
    count = 0
    total_psnr = 0
    total_bitrate = 0
    total_time = 0
    
    # Process each image
    for sample, ref_samples, key in test_dataset:
        x = sample.unsqueeze(0).to(device)
        
        # Resize reference images to match x's dimensions BEFORE sending to device
        ref_samples = [r.unsqueeze(0) for r in ref_samples]  # Add batch dimension first
        ref_samples = resize_reference_images(ref_samples, x.size())
        ref_samples = [r.to(device) for r in ref_samples]  # Now send to device
        
        # Print dimensions for verification
        # print(f"Input image dimensions: {x.shape}")
        # for i, ref in enumerate(ref_samples):
        #     print(f"Reference {i+1} dimensions after resize: {ref.shape}")
        
        x_padded, padding = pad(x, p)
        ref_samples_padded = [pad(r, p)[0] for r in ref_samples]
        count += 1
        
        with torch.no_grad():
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            start_time = time.time()
            
            # Compress and decompress
            out_enc = net.compress(x_padded, ref_samples_padded)
            out_dec = net.decompress(out_enc["strings"], out_enc["shape"], ref_samples_padded)
                
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            end_time = time.time()
            
            # Process results
            total_time += (end_time - start_time)
            out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
            num_pixels = x.size(0) * x.size(2) * x.size(3)
            
            # Calculate metrics
            bitrate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
            psnr = compute_psnr(x, out_dec["x_hat"])
            
            # Print individual results
            print(f'{key} Bitrate: {bitrate:.3f}bpp')
            print(f'{key} PSNR: {psnr:.2f}dB')
            
            # Accumulate metrics
            total_bitrate += bitrate
            total_psnr += psnr
    
    # Calculate averages
    avg_psnr = total_psnr / count
    avg_bitrate = total_bitrate / count
    avg_time = total_time / count
    
    print(f'Average PSNR: {avg_psnr:.2f}dB')
    print(f'Average Bitrate: {avg_bitrate:.3f} bpp')
    print(f'Average Time: {avg_time:.3f} s')
    
    return {
        'checkpoint': checkpoint_path,
        'bitrate': avg_bitrate,
        'psnr': avg_psnr,
        'time': avg_time
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints and plot RD curve.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--data", type=str, default='/h3cstore_ns/ydchen/DATASET/kodak.hdf5',
                        help="Path to dataset")
    parser.add_argument("--ref_path", type=str, default='/h3cstore_ns/ydchen/DATASET/coding_img_cropped_2/Flickr2K.hdf5',
                        help="Path to reference dataset for CLC")
    parser.add_argument("--feature_cache_path", type=str, default='/h3cstore_ns/ydchen/code/CompressAI/LIC_TCM/model_ckpt_TCM/data_cluster_feature/flicker_features.pkl',
                        help="Path to feature cache for CLC")
    parser.add_argument("--models_dir", type=str, default='/h3cstore_ns/ydchen/code/CompressAI/LIC_TCM/CLC_models',
                        help="Directory containing checkpoint folders")
    parser.add_argument("--n_clusters", type=int, default=3000, help="Number of clusters for CLC")
    parser.add_argument("--n_refs", type=int, default=3, help="Number of reference images for CLC")
    args = parser.parse_args()
    
    # Set device
    device = 'cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create results directory
    results_dir = get_results_dir(args.models_dir)
    print(f"Results will be saved to: {results_dir}")
    
    # Get all checkpoints to evaluate
    checkpoints = get_all_checkpoints(args.models_dir)
    print(f"Found {len(checkpoints)} checkpoints to evaluate:")
    for i, cp in enumerate(checkpoints, 1):
        print(f"{i}. Bitrate point: {cp['bitrate']}, Path: {cp['path']}")
    
    # CSV to store results
    csv_path = os.path.join(results_dir, 'rd_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Checkpoint', 'Bitrate (bpp)', 'PSNR (dB)', 'Time (s)'])
        
        # Evaluate each checkpoint
        results = []
        for i, cp in enumerate(checkpoints, 1):
            print(f"\n[{i}/{len(checkpoints)}] Evaluating checkpoint for bitrate {cp['bitrate']}...")
            
            # Evaluate the checkpoint
            result = evaluate_checkpoint(
                cp['path'], 
                args.data, 
                args.ref_path, 
                args.feature_cache_path, 
                device,
                args.n_clusters,
                args.n_refs
            )
            
            # Store result
            results.append(result)
            
            # Write to CSV
            csvwriter.writerow([
                cp['path'],
                f"{result['bitrate']:.4f}",
                f"{result['psnr']:.2f}",
                f"{result['time']:.4f}"
            ])
            
            # Flush to save progress
            csvfile.flush()
            
            print(f"Completed {i}/{len(checkpoints)} checkpoints")
    
    # Plot the RD curve
    rd_curve_path = os.path.join(results_dir, 'rd_curve.png')
    plot_rd_curve(results, rd_curve_path)
    
    # Print final summary
    print("\nEvaluation complete!")
    print(f"Results saved to: {csv_path}")
    print(f"RD curve saved to: {rd_curve_path}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available, using CPU")
    main()
