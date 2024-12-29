import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableAlignment(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.offset_conv = nn.Conv2d(input_dim*2, 2*3*3, kernel_size=3, padding=1)
        self.modulation_conv = nn.Conv2d(input_dim*2, 3*3, kernel_size=3, padding=1)
        
    def forward(self, x, similarity_map):
        B, C, H, W = x.shape
        
        sim_reshaped = similarity_map.view(B, H, W, H, W)
        
        weighted_x = torch.zeros_like(x)
        for i in range(H):
            for j in range(W):
                weights = sim_reshaped[:, i, j, :, :].view(B, 1, H, W)
                weighted_x += weights * x
                
        concat_feat = torch.cat([x, weighted_x], dim=1)
        
        offset = self.offset_conv(concat_feat)
        modulation = torch.sigmoid(self.modulation_conv(concat_feat))
        
        n = 3 * 3
        offset = offset.view(B, n, 2, H, W)
        modulation = modulation.view(B, n, 1, H, W)
        
        result = self.deform_conv(x, offset, modulation)
        
        return result
    
    def deform_conv(self, x, offset, modulation):
        B, C, H, W = x.shape
        result = torch.zeros_like(x)
        
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    for k in range(9):
                        off_h = h + offset[b, k, 0, h, w]
                        off_w = w + offset[b, k, 1, h, w]
                        
                        if (0 <= off_h <= H-1) and (0 <= off_w <= W-1):
                            h0, w0 = int(off_h), int(off_w)
                            h1, w1 = min(h0 + 1, H-1), min(w0 + 1, W-1)
                            
                            lambda_h = off_h - h0
                            lambda_w = off_w - w0
                            
                            val = (1-lambda_h)*(1-lambda_w)*x[b,:,h0,w0] + \
                                  lambda_h*(1-lambda_w)*x[b,:,h1,w0] + \
                                  (1-lambda_h)*lambda_w*x[b,:,h0,w1] + \
                                  lambda_h*lambda_w*x[b,:,h1,w1]
                            
                            result[b,:,h,w] += val * modulation[b,k,0,h,w]
                            
        return result

class CLM(nn.Module):
    """Conditional Latent Matching module"""
    def __init__(self, input_dim, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.feature_transform = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, input_dim, 1)
        )
        self.alignment = DeformableAlignment(input_dim)
        
        # 添加注意力融合层
        self.attention_conv = nn.Conv2d(input_dim, 1, 1)
        
        # 添加最终融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, input_dim, 3, padding=1)
        )
        
    def forward(self, y, y_refs):
        """
        Args:
            y: Input latent tensor [B, C, H, W]
            y_refs: List of reference latent tensors [M, B, C, H, W]
        Returns:
            fused: Fused tensor [B, C, H, W]
        """
        B, C, H, W = y.shape
        M = len(y_refs)
        
        # Transform features
        y_t = self.feature_transform(y)
        y_refs_t = [self.feature_transform(y_ref) for y_ref in y_refs]
        
        # Calculate similarity matrix and align each reference
        aligned_features = []
        attention_weights = []
        
        for i, (y_ref, y_ref_t) in enumerate(zip(y_refs, y_refs_t)):
            # Calculate similarity
            sim = torch.bmm(y_t.view(B, C, -1).transpose(1, 2),
                          y_ref_t.view(B, C, -1)) / self.temperature
            similarity_map = F.softmax(sim, dim=-1)
            
            # Align features
            aligned = self.alignment(y_ref, similarity_map)
            aligned_features.append(aligned)
            
            # Calculate attention weight
            attention = self.attention_conv(aligned)
            attention_weights.append(attention)
        
        # Stack attention weights and apply softmax
        attention_weights = torch.stack(attention_weights, dim=1)  # [B, M, 1, H, W]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum of aligned features
        aligned_stack = torch.stack(aligned_features, dim=1)  # [B, M, C, H, W]
        weighted_sum = (aligned_stack * attention_weights).sum(dim=1)  # [B, C, H, W]
        
        # Final fusion with input feature
        fused = self.fusion_conv(weighted_sum + y)
        
        return fused
    
class SimpleCLM(nn.Module):
    """Simplified Conditional Latent Matching module"""
    def __init__(self, input_dim, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
        # 简化特征变换层
        self.feature_transform = nn.Conv2d(input_dim, input_dim, 1)
        
        # 简化注意力层
        self.attention_conv = nn.Conv2d(input_dim, 1, 1)
        
        # 简化最终融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, y, y_refs):
        """
        Args:
            y: Input latent tensor [B, C, H, W]
            y_refs: List of reference latent tensors [M, B, C, H, W]
        Returns:
            fused: Fused tensor [B, C, H, W]
        """
        B, C, H, W = y.shape
        M = len(y_refs)
        
        # 特征变换
        y_t = self.feature_transform(y)
        
        aligned_features = []
        attention_weights = []
        
        for y_ref in y_refs:
            # 简化的特征对齐：直接使用卷积变换
            ref_t = self.feature_transform(y_ref)
            
            # 计算相似度并生成注意力图
            attention = self.attention_conv(ref_t)
            attention_weights.append(attention)
            
            # 使用注意力加权的特征
            weighted_feat = ref_t * torch.sigmoid(attention)
            aligned_features.append(weighted_feat)
        
        # 堆叠并融合特征
        attention_weights = torch.stack(attention_weights, dim=1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        aligned_stack = torch.stack(aligned_features, dim=1)
        weighted_sum = (aligned_stack * attention_weights).sum(dim=1)
        
        # 最终融合
        fused = self.fusion_conv(weighted_sum + y)
        
        return fused

if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    
    # 测试参数
    batch_size = 2
    channels = 64
    height = 32
    width = 32
    num_refs = 3
    temperature = 0.5
    
    # 创建模型
    clm = CLM(input_dim=channels, temperature=temperature)
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'    
    # 创建测试数据
    y = torch.randn(batch_size, channels, height, width)
    y_refs = [torch.randn(batch_size, channels, height, width) for _ in range(num_refs)]
    clm.to(device)
    y = y.to(device)
    y_refs = [y_ref.to(device) for y_ref in y_refs]
    # 运行模型
    fused = clm(y, y_refs)


    
    # 打印结果
    print(f"Input shape: {y.shape}")
    print(f"Number of reference frames: {len(y_refs)}")
    print(f"Reference frame shape: {y_refs[0].shape}")
    print(f"Fused output shape: {fused.shape}")
    
    # 验证输出
    print("\nValidation:")
    print(f"Output shape matches input: {fused.shape == y.shape}")
    
    # 计算统计信息
    mean_diff = torch.mean(torch.abs(fused - y))
    print(f"\nMean absolute difference from input: {mean_diff:.4f}")
    print(f"Fused features range: [{fused.min():.4f}, {fused.max():.4f}]")
    print(f"Input features range: [{y.min():.4f}, {y.max():.4f}]")