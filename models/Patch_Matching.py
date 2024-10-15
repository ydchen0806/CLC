import pdb
import time
from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pylab

import torch
from torch import nn
import torch.nn.functional as F

from compressai.utils import conv, deconv
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai_local.models.google import JointAutoregressiveHierarchicalPriors
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    GDN,
)

def show(tensor, title=None):
    img = tensor.cpu().numpy().squeeze(0)
    plt.imshow(img)
    pylab.show()

'''
def SI_Finder_at_Image_Domain(x_dec, y_imgs, patch_h, patch_w, y_dec, mask=None):
    
    N, C, H, W = x_dec.shape
    _, _, img_h, img_w = y_imgs.shape

    # Transpose to channel last
    x_dec_t = x_dec
    y_imgs_t = y_imgs
    y_dec_t = y_dec
    #pdb.set_trace()
    for n in range(N):
        x_img_dec = x_dec_t[n].unsqueeze(0)
        y_img = y_imgs_t[n].unsqueeze(0)
        y_img_dec = y_dec_t[n].unsqueeze(0)

        # Extract patches
        x_patches = x_img_dec.reshape(1, C, H//patch_h, patch_h, W//patch_w, patch_w).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, patch_h, patch_w)
        patchs_num = x_patches.shape[0]
        q = rgb_transform(reduce_mean_and_std_normalize_images(x_patches*255))
        r = rgb_transform(reduce_mean_and_std_normalize_images(y_img_dec*255))
        
        img = y_img_dec.detach().cpu().numpy().squeeze(0).transpose(1,2,0)
        plt.imshow(img)
        plt.show()
        pylab.show()
        
        img = r.detach().cpu().numpy().squeeze(0).transpose(1,2,0)
        plt.imshow(img)
        plt.show()
        pylab.show()


        cross_corr = L2_or_pearson_corr(q, r, patch_h, patch_w) * mask
        
        _, _, corr_h, corr_w = cross_corr.shape
        cross_corr = cross_corr.reshape(1, -1, corr_h*corr_w)
        index = torch.argmax(cross_corr, dim=2).squeeze(0)
        index_h, index_w = torch.div(index, corr_w, rounding_mode='floor'), index % corr_w   #[80]
        
        patch_h_index, patch_w_index = torch.meshgrid(torch.arange(0, patch_h).cuda(), torch.arange(0, patch_w).cuda())  #[80,48]
        
        
        index_h_to_patch, index_w_to_patch = index_h.unsqueeze(1).unsqueeze(1) + patch_h_index, index_w.unsqueeze(1).unsqueeze(1) + patch_w_index
        
        pixel_index = (index_h_to_patch * img_w + index_w_to_patch).reshape(-1)
        
        y_img_patches = torch.index_select(y_img.reshape(-1, C, img_h*img_w), 2, pixel_index).reshape(-1, C, patchs_num, patch_h, patch_w)
        y_img_reference = y_img_patches.reshape(-1, C, img_h//patch_h, img_w//patch_w, patch_h, patch_w).permute(0, 1, 2, 4, 3, 5).reshape(-1, C, img_h, img_w)
        if n==0:
            y_img_references = y_img_reference
        else:
            y_img_references = torch.concat([y_img_references, y_img_reference], axis=0)

    return y_img_references  
 '''
def SI_Finder_at_Image_Domain(x_dec, y_imgs, patch_h, patch_w, y_dec, mask=None, is_pearson_corr_cpu=False):
    
    N, C, H, W = x_dec.shape
    _, _, img_h, img_w = y_imgs.shape

    # Transpose to channel last
    x_dec_t = x_dec
    y_imgs_t = y_imgs
    y_dec_t = y_dec

    for n in range(N):
        x_img_dec = x_dec_t[n].unsqueeze(0)
        y_img = y_imgs_t[n].unsqueeze(0)
        y_img_dec = y_dec_t[n].unsqueeze(0)

        # Extract patches
        x_patches = x_img_dec.reshape(1, C, H//patch_h, patch_h, W//patch_w, patch_w).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, patch_h, patch_w)
        patchs_num = x_patches.shape[0]
        q = rgb_transform(reduce_mean_and_std_normalize_images(x_patches*255))
        r = rgb_transform(reduce_mean_and_std_normalize_images(y_img_dec*255))
        cross_corr = L2_or_pearson_corr(q, r, patch_h, patch_w, is_cpu=is_pearson_corr_cpu) * mask
        _, _, corr_h, corr_w = cross_corr.shape
        cross_corr = cross_corr.reshape(1, -1, corr_h*corr_w)
        index = torch.argmax(cross_corr, dim=2).squeeze(0)
        index_h, index_w = torch.div(index, corr_w, rounding_mode='floor'), index % corr_w
        patch_h_index, patch_w_index = torch.meshgrid(torch.arange(0, patch_h).cuda(), torch.arange(0, patch_w).cuda())
        index_h_to_patch, index_w_to_patch = index_h.unsqueeze(1).unsqueeze(1) + patch_h_index, index_w.unsqueeze(1).unsqueeze(1) + patch_w_index
        pixel_index = (index_h_to_patch * img_w + index_w_to_patch).reshape(-1)
        y_img_patches = torch.index_select(y_img.reshape(-1, C, img_h*img_w), 2, pixel_index).reshape(-1, C, patchs_num, patch_h, patch_w)
        y_img_reference = y_img_patches.reshape(-1, C, img_h//patch_h, img_w//patch_w, patch_h, patch_w).permute(0, 1, 2, 4, 3, 5).reshape(-1, C, img_h, img_w)
        if n==0:
            y_img_references = y_img_reference
        else:
            y_img_references = torch.concat([y_img_references, y_img_reference], axis=0)

    return y_img_references  

def SI_Finder_at_Vgg19_Feature_Domain(x_decs, ys, patch_h, patch_w, y_decs, mask=None):
    N, C, H, W = x_decs.shape
    _, _, feature_h, feature_w = ys.shape

    for n in range(N):
        x_dec = x_decs[n].unsqueeze(0)
        y = ys[n].unsqueeze(0)
        y_dec = y_decs[n].unsqueeze(0)

        # Extract patches
        x_dec_patches = x_dec.reshape(1, C, H//patch_h, patch_h, W//patch_w, patch_w).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, patch_h, patch_w)
        patchs_num = x_dec_patches.shape[0]
        q = x_dec_patches
        r = y_dec

        cross_corr = L2_or_pearson_corr(q, r, patch_h, patch_w) * mask
        pdb.set_trace()
        _, _, corr_h, corr_w = cross_corr.shape
        cross_corr = cross_corr.reshape(1, -1, corr_h*corr_w)
        index = torch.argmax(cross_corr, dim=2).squeeze(0)
        index_h, index_w = torch.div(index, corr_w, rounding_mode='floor'), index % corr_w
        patch_h_index, patch_w_index = torch.meshgrid(torch.arange(0, patch_h).cuda(), torch.arange(0, patch_w).cuda())
        index_h_to_patch, index_w_to_patch = index_h.unsqueeze(1).unsqueeze(1) + patch_h_index, index_w.unsqueeze(1).unsqueeze(1) + patch_w_index
        pixel_index = (index_h_to_patch * feature_w + index_w_to_patch).reshape(-1)
        y_patches = torch.index_select(y.reshape(-1, C, feature_h*feature_w), 2, pixel_index).reshape(-1, C, patchs_num, patch_h, patch_w)
        y_reference = y_patches.reshape(-1, C, feature_h//patch_h, feature_w//patch_w, patch_h, patch_w).permute(0, 1, 2, 4, 3, 5).reshape(-1, C, feature_h, feature_w)
        if n==0:
            y_references = y_reference
        else:
            y_references = torch.concat([y_references, y_reference], axis=0)

    return y_references    

def SI_Finder_at_Decoder_Feature_Domain(x_decs, ys, patch_h, patch_w, y_decs, layer_names, args, mask=None, other_ys=None, is_img_patch_matching=False, is_pearson_corr_cpu=False):
    N, C, H, W = x_decs.shape
    _, _, feature_h, feature_w = ys.shape

    for n in range(N):
        y_reference = {}
        x_dec = x_decs[n].unsqueeze(0)
        y = ys[n].unsqueeze(0)
        y_dec = y_decs[n].unsqueeze(0)
        if other_ys is not None:
            other_y = [other_y_i[n].unsqueeze(0) for other_y_i in other_ys]
        else:
            other_y = None    

        # Extract patches
        x_dec_patches = x_dec.reshape(1, C, H // patch_h, patch_h, W // patch_w, patch_w).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, patch_h, patch_w)
        patches_num = x_dec_patches.shape[0]
        if not is_img_patch_matching:
            q = x_dec_patches
            r = y_dec
        else:
            q = rgb_transform(reduce_mean_and_std_normalize_images(x_dec_patches * 255))
            r = rgb_transform(reduce_mean_and_std_normalize_images(y_dec * 255))

        cross_corr = L2_or_pearson_corr(q, r, patch_h, patch_w, is_cpu=is_pearson_corr_cpu)
        if mask is not None:
            cross_corr = cross_corr * mask
            
        y_reference[layer_names[0]] = SI_Wraper(cross_corr, patch_h, patch_w, patches_num, y, args.num_k, args.temperature, args.is_stack)

        if args.single_layer != 0:
            if args.single_layer == 4:
                pass
            else:
                if args.single_layer == 3:
                    i = 0
                elif args.single_layer == 2:
                    i = 1
                elif args.single_layer == 1:
                    i = 2
                other_y_i = other_y[i]
                _, _, feature_h_i, feature_w_i = other_y_i.shape
                assert (feature_h // feature_h_i == 2 ** (i + 1)) and (feature_w // feature_w_i == 2 ** (i + 1))    
                cross_corr_i = cross_corr[:, :, ::2 ** (i + 1), ::2 ** (i + 1)]   
                y_reference[str(4 - i)] = SI_Wraper(cross_corr_i, patch_h // 2 ** (i + 1), patch_w // 2 ** (i + 1), patches_num, other_y_i, args.num_k, args.temperature, args.is_stack)
        else:       
            if other_y is not None:
                for i, other_y_i in enumerate(other_y):
                    _, _, feature_h_i, feature_w_i = other_y_i.shape
                    assert (feature_h // feature_h_i == 2 ** (i + 1)) and (feature_w // feature_w_i == 2 ** (i + 1))    
                    cross_corr_i = cross_corr[:, :, ::2 ** (i + 1), ::2 ** (i + 1)]   
                    y_reference[layer_names[i + 1]] = SI_Wraper(cross_corr_i, patch_h // 2 ** (i + 1), patch_w // 2 ** (i + 1), patches_num, other_y_i, args.num_k, args.temperature, args.is_stack)
        
        if n == 0:
            y_references = y_reference
        else:
            for key in y_references.keys():
                y_references[key] = torch.concat([y_references[key], y_reference[key]], axis=0)

    return y_references  

def SI_Wraper(cross_corr, patch_h, patch_w, patchs_num, y, k = 1, temperature=15, is_stack=False):

    _, _, corr_h, corr_w = cross_corr.shape
    _, C, feature_h, feature_w = y.shape
    cross_corr = cross_corr.reshape(1, -1, corr_h*corr_w)

    value, index = torch.topk(cross_corr, k, dim=2)
    value, index = value.squeeze(0), index.squeeze(0)
    weight = F.softmax(value*temperature, dim=1)

    index_h, index_w = torch.div(index, corr_w, rounding_mode='floor'), index % corr_w
    patch_h_index, patch_w_index = torch.meshgrid(torch.arange(0, patch_h).cuda(), torch.arange(0, patch_w).cuda())
    index_h_to_patch, index_w_to_patch = index_h.unsqueeze(2).unsqueeze(2) + patch_h_index, index_w.unsqueeze(2).unsqueeze(2) + patch_w_index
    pixel_index = (index_h_to_patch * feature_w + index_w_to_patch).reshape(-1)
    y_patches = torch.index_select(y.reshape(-1, C, feature_h*feature_w), 2, pixel_index).reshape(-1, C, patchs_num, k, patch_h, patch_w)

    if is_stack:
        y_reference = y_patches.reshape(-1, C, feature_h//patch_h, feature_w//patch_w, k, patch_h, patch_w).permute(0, 4, 1, 2, 5, 3, 6).reshape(-1, k*C, feature_h, feature_w)
    else:
        y_patches = torch.sum(y_patches * weight.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1), 3)
        y_reference = y_patches.reshape(-1, C, feature_h//patch_h, feature_w//patch_w, patch_h, patch_w).permute(0, 1, 2, 4, 3, 5).reshape(-1, C, feature_h, feature_w)

    return y_reference  


class SiNet(nn.Module):

    def __init__(self, N=192, **kwargs):
        super().__init__(**kwargs)

        self.sinet = nn.Sequential(
            ResidualBlock(6, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, 3),
        )     

    def forward(self, x):
        return {"x_hat": self.sinet(x)+x[:, :3, :, :],}   

class SiNet2(nn.Module):

    def __init__(self, N=192, **kwargs):
        super().__init__(**kwargs)

        self.sinet = nn.Sequential(
            nn.Conv2d(6, N, (3,3), dilation=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, (3,3), dilation=2, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, (3,3), dilation=4, padding=4, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, (3,3), dilation=8, padding=8, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, (3,3), dilation=16, padding=16, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, (3,3), dilation=32, padding=32, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, (3,3), dilation=64, padding=64, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, (3,3), dilation=128, padding=128, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, (3,3), dilation=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(N, 3, (1,1), dilation=1, padding=0),
        )     

    def forward(self, x):
        return {"x_hat": self.sinet(x),}        

class Cheng2020Anchor_Encoder(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=128, **kwargs):
        super().__init__(N=N, M=N, **kwargs)
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )
        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
    def forward(self, x):
        x_height, x_width = x.size()[2:4]
        x_res_h, x_res_w = (16 - x_height%16)%16, (16 - x_width%16)%16
        paddings = (0, x_res_w, 0, x_res_h)
        x = F.pad(x, paddings, "replicate")
        y = self.g_a(x)
        y_height, y_width = y.size()[2:4]
        y_res_h, y_res_w =(4 - y_height%4)%4, (4 - y_width%4)%4
        paddings = (0, y_res_w, 0, y_res_h)
        y = F.pad(y, paddings, "replicate")
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)[:, :, :y_height, :y_width]

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)[:, :, :y_height, :y_width]
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat = y_hat[:, :, :y_height, :y_width]
        _, y_likelihoods = self.gaussian_conditional(y_hat, scales_hat, means=means_hat, is_quant=False)
        
        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

class Cheng2020Anchor_Decoderx2(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=128, is_skip_connect=True, **kwargs):
        super().__init__(N=N, M=N, **kwargs)
        self.g_s = nn.Sequential(
            ResidualBlock(N*2, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )
        self.N = int(N)
        self.is_skip_connect = is_skip_connect
        #self.x_height = x_height
        #self.x_width = x_width
    def forward(self, out_net1, x, y_hat):
        #pdb.set_trace()
        x_height, x_width = x.size()[2:4]
        y_hatx2 = self.g_s._modules['1'](self.g_s._modules['0'](y_hat))
        y_hatx4 = self.g_s._modules['3'](self.g_s._modules['2'](y_hatx2))
        y_hatx8 = self.g_s._modules['5'](self.g_s._modules['4'](y_hatx4))
        x_hat = self.g_s._modules['7'](self.g_s._modules['6'](y_hatx8))[:, :, :x_height, :x_width]
        if self.is_skip_connect:
            x_hat = x_hat + out_net1['x_hat']
        return {"x_hat": x_hat,}
        
class Conditional_Texture_Transfer_Network(nn.Module):

    def __init__(self, N=192, M=192, is_rb =False, is_skip_connect=True, is_skip_connect_in_feature_domain=False, **kwargs):
        super().__init__(**kwargs)
        self.is_rb = is_rb
        self.is_skip_connect = is_skip_connect
        self.is_skip_connect_in_feature_domain = is_skip_connect_in_feature_domain
        if not is_rb:
            self.network1 = nn.Sequential(deconv(M, N, kernel_size=5, stride=2),
                                         GDN(N, inverse=True),)
            self.network2 = nn.Sequential(deconv(N, N, kernel_size=5, stride=2),
                                         GDN(N, inverse=True),)
            self.network3 = nn.Sequential(deconv(N+256, N, kernel_size=5, stride=2),
                                         GDN(N, inverse=True),)
            self.network4 = nn.Sequential(deconv(N+128, N, kernel_size=5, stride=2))
            self.network5 = nn.Sequential(conv(N+64, 3, kernel_size=5, stride=1),)
                                                                      
        else:    
            self.network1 = nn.Sequential(ResidualBlock(M, N),
                                     ResidualBlockUpsample(N, N, 2),)  
            self.network2 = nn.Sequential(ResidualBlock(N, N),
                                     ResidualBlockUpsample(N, N, 2),)
            self.network3 = nn.Sequential(ResidualBlock(N+256, N),
                                     ResidualBlockUpsample(N, N, 2),)
            self.network4 = nn.Sequential(ResidualBlock(N+128, N),
                                     subpel_conv3x3(N, N, 2),) 
            self.network5 = nn.Sequential(ResidualBlock(N+64, 3),)                         

    def forward(self, x, reference_features, vgg19_select_layernames, first_rec):  
        x = self.network1(x) 
        x = self.network2(x)             
        vgg19_select_layernames = vgg19_select_layernames.split('/')

        if self.is_skip_connect_in_feature_domain:
            x = self.network3._modules['1'](self.network3._modules['0'](torch.concat((x, reference_features[vgg19_select_layernames[0]]), 1)) + x)
            x = self.network4._modules['1'](self.network4._modules['0'](torch.concat((x, reference_features[vgg19_select_layernames[1]]), 1)) + x)
            x = self.network5(torch.concat((x, reference_features[vgg19_select_layernames[2]]), 1))
        else:    
            x = self.network3(torch.concat((x, reference_features[vgg19_select_layernames[0]]), 1))
            x = self.network4(torch.concat((x, reference_features[vgg19_select_layernames[1]]), 1)) 
            x = self.network5(torch.concat((x, reference_features[vgg19_select_layernames[2]]), 1))

        if self.is_skip_connect:
            return {"x_hat": x+first_rec,}     
        else:
            return {"x_hat": x,}    

class Encoder_based_Conditional_Texture_Transfer_Network(nn.Module):

    def __init__(self, N=192, M=192, args=None, is_patch_matching_in_img_domain=True, is_skip_connect=True, is_skip_connect_in_feature_domain=False, is_not_use_si=False, **kwargs):

        super().__init__(**kwargs)
        
        self.is_patch_matching_in_img_domain = is_patch_matching_in_img_domain
        self.is_skip_connect = is_skip_connect
        self.is_skip_connect_in_feature_domain = is_skip_connect_in_feature_domain
        if args.is_stack:
            ref_N = args.num_k * N
        else:
            ref_N = N           
        if args.single_layer != 0:
            if args.single_layer == 1:
                self.network1 = nn.Sequential(ResidualBlock(M+ref_N, N),
                                             ResidualBlockUpsample(N, N, 2),)
            else:
                self.network1 = nn.Sequential(ResidualBlock(M, N),
                                             ResidualBlockUpsample(N, N, 2),)
            if args.single_layer == 2:
                self.network2 = nn.Sequential(ResidualBlock(2*N+ref_N, N),
                                             ResidualBlockUpsample(N, N, 2),)
            else:
                self.network2 = nn.Sequential(ResidualBlock(2*N, N),
                                             ResidualBlockUpsample(N, N, 2),)
            if args.single_layer == 3:
                self.network3 = nn.Sequential(ResidualBlock(2*N+ref_N, N),
                                             ResidualBlockUpsample(N, N, 2),)
            else:
                self.network3 = nn.Sequential(ResidualBlock(2*N, N),
                                         ResidualBlockUpsample(N, N, 2),)
            if args.single_layer == 4:
                self.network4 = nn.Sequential(ResidualBlock(2*N+ref_N, N),
                                             subpel_conv3x3(N, N, 2),)
            else:
                self.network4 = nn.Sequential(ResidualBlock(2*N, N),
                                             subpel_conv3x3(N, N, 2),) 
            self.network5 = nn.Sequential(ResidualBlock(N, 3),)
        elif args.is_not_use_si:
            self.network1 = nn.Sequential(ResidualBlock(M, N),
                                         ResidualBlockUpsample(N, N, 2),)  
            self.network2 = nn.Sequential(ResidualBlock(2*N, N),
                                         ResidualBlockUpsample(N, N, 2),)
            self.network3 = nn.Sequential(ResidualBlock(2*N, N),
                                         ResidualBlockUpsample(N, N, 2),)
            self.network4 = nn.Sequential(ResidualBlock(2*N, N),
                                         subpel_conv3x3(N, N, 2),) 
            self.network5 = nn.Sequential(ResidualBlock(N, 3),)
        else:
            self.network1 = nn.Sequential(ResidualBlock(M+ref_N, N),
                                         ResidualBlockUpsample(N, N, 2),)  
            self.network2 = nn.Sequential(ResidualBlock(2*N+ref_N, N),
                                         ResidualBlockUpsample(N, N, 2),)
            self.network3 = nn.Sequential(ResidualBlock(2*N+ref_N, N),
                                         ResidualBlockUpsample(N, N, 2),)
            self.network4 = nn.Sequential(ResidualBlock(2*N+ref_N, N),
                                         subpel_conv3x3(N, N, 2),) 
            if self.is_patch_matching_in_img_domain:                             
                self.network5 = nn.Sequential(ResidualBlock(2*N, 3),)   
            else:                          
                self.network5 = nn.Sequential(ResidualBlock(N, 3),)    

    def forward(self, net_out, args=None, reference_features=None):  

        if args.single_layer != 0:
            if args.single_layer == 1:
                if self.is_skip_connect_in_feature_domain:
                    net = self.network1._modules['1'](self.network1._modules['0'](torch.concat((net_out['y_hat'], reference_features['1']), 1)))
                else:
                    net = self.network1._modules['1'](self.network1._modules['0'](torch.concat((net_out['y_hat'], reference_features['1']), 1)))
            else:
                if self.is_skip_connect_in_feature_domain:
                    net = self.network1._modules['1'](self.network1._modules['0'](net_out['y_hat']))
                else:
                    net = self.network1._modules['1'](self.network1._modules['0'](net_out['y_hat']))
            if args.single_layer == 2:
                if self.is_skip_connect_in_feature_domain:
                    net = self.network2._modules['1'](self.network2._modules['0'](torch.concat((net_out['y_hatx2'], reference_features['2'], net), 1)) + net_out['y_hatx2'])
                else:
                    net = self.network2._modules['1'](self.network2._modules['0'](torch.concat((net_out['y_hatx2'], reference_features['2'], net), 1)))
            else:
                if self.is_skip_connect_in_feature_domain:
                    net = self.network2._modules['1'](self.network2._modules['0'](torch.concat((net_out['y_hatx2'], net), 1)) + net_out['y_hatx2'])
                else:
                    net = self.network2._modules['1'](self.network2._modules['0'](torch.concat((net_out['y_hatx2'], net), 1)))
            if args.single_layer == 3:
                if self.is_skip_connect_in_feature_domain:
                    net = self.network3._modules['1'](self.network3._modules['0'](torch.concat((net_out['y_hatx4'], reference_features['4'], net), 1)) + net_out['y_hatx4'])
                else:
                    net = self.network3._modules['1'](self.network3._modules['0'](torch.concat((net_out['y_hatx4'], reference_features['4'], net), 1)))
            else:
                if self.is_skip_connect_in_feature_domain:
                    net = self.network3._modules['1'](self.network3._modules['0'](torch.concat((net_out['y_hatx4'], net), 1)) + net_out['y_hatx4'])
                else:
                    net = self.network3._modules['1'](self.network3._modules['0'](torch.concat((net_out['y_hatx4'], net), 1)))
            if args.single_layer == 4:
                if self.is_skip_connect_in_feature_domain:
                    net = self.network4._modules['1'](self.network4._modules['0'](torch.concat((net_out['y_hatx8'], reference_features['8'], net), 1)) + net_out['y_hatx8'])
                else:
                    net = self.network4._modules['1'](self.network4._modules['0'](torch.concat((net_out['y_hatx8'], reference_features['8'], net), 1)))
            else:
                if self.is_skip_connect_in_feature_domain:
                    net = self.network4._modules['1'](self.network4._modules['0'](torch.concat((net_out['y_hatx8'], net), 1)) + net_out['y_hatx8'])
                else:
                    net = self.network4._modules['1'](self.network4._modules['0'](torch.concat((net_out['y_hatx8'], net), 1)))
            x = self.network5(net)
        elif args.is_not_use_si:
            #pdb.set_trace()
            if self.is_skip_connect_in_feature_domain:
                net = self.network1._modules['1'](self.network1._modules['0'](net_out['y_hat']))
                net = self.network2._modules['1'](self.network2._modules['0'](torch.concat((net_out['y_hatx2'], net), 1)) + net_out['y_hatx2'])
                net = self.network3._modules['1'](self.network3._modules['0'](torch.concat((net_out['y_hatx4'], net), 1)) + net_out['y_hatx4'])
                net = self.network4._modules['1'](self.network4._modules['0'](torch.concat((net_out['y_hatx8'], net), 1)) + net_out['y_hatx8'])
            else:    
                net = self.network1._modules['1'](self.network1._modules['0'](net_out['y_hat']))
                net = self.network2._modules['1'](self.network2._modules['0'](torch.concat((net_out['y_hatx2'], net), 1)))
                net = self.network3._modules['1'](self.network3._modules['0'](torch.concat((net_out['y_hatx4'], net), 1)))
                net = self.network4._modules['1'](self.network4._modules['0'](torch.concat((net_out['y_hatx8'], net), 1)))
                
            x = self.network5(net)    
            
        else: 
            if self.is_skip_connect_in_feature_domain:
                net = self.network1._modules['1'](self.network1._modules['0'](torch.concat((net_out['y_hat'], reference_features['1']), 1)))
                net = self.network2._modules['1'](self.network2._modules['0'](torch.concat((net_out['y_hatx2'], reference_features['2'], net), 1)) + net_out['y_hatx2'])
                net = self.network3._modules['1'](self.network3._modules['0'](torch.concat((net_out['y_hatx4'], reference_features['4'], net), 1)) + net_out['y_hatx4'])
                net = self.network4._modules['1'](self.network4._modules['0'](torch.concat((net_out['y_hatx8'], reference_features['8'], net), 1)) + net_out['y_hatx8'])
            else:    
                net = self.network1._modules['1'](self.network1._modules['0'](torch.concat((net_out['y_hat'], reference_features['1']), 1)))
                net = self.network2._modules['1'](self.network2._modules['0'](torch.concat((net_out['y_hatx2'], reference_features['2'], net), 1)))
                net = self.network3._modules['1'](self.network3._modules['0'](torch.concat((net_out['y_hatx4'], reference_features['4'], net), 1)))
                net = self.network4._modules['1'](self.network4._modules['0'](torch.concat((net_out['y_hatx8'], reference_features['8'], net), 1)))
            
            if self.is_patch_matching_in_img_domain:
                x = self.network5(torch.concat((net, reference_features['16']), 1))
            else:
                x = self.network5(net)

        if self.is_skip_connect:
            return {"x_hat": x+net_out['x_hat'],}     
        else:
            return {"x_hat": x,}   

class Non_Local_Network(nn.Module):

    def __init__(self, N=192, **kwargs):

        super().__init__(**kwargs)
        
        self.theta = nn.Sequential(ResidualBlock(N, N),)  
        self.phi = nn.Sequential(ResidualBlock(N, N),)
        self.avgpool = nn.AvgPool2d(2, stride=2)
 

    def forward(self, feature_dec1, feature_dec2, reference_feature, layer_names, mask=None, other_reference_feature=None, is_detach=True):  
        reference_features = {}
        N, C, H, W = feature_dec1.shape
        if is_detach:
            f1 = self.theta(feature_dec1.detach())
            f2 = self.phi(feature_dec2.detach())
        else:
            f1 = self.theta(feature_dec1)
            f2 = self.phi(feature_dec2)    

        f1 = f1.reshape(N, C, H*W).permute(0, 2, 1)
        f2 = f2.reshape(N, C, H*W)

        corr = torch.matmul(f1, f2)
        if mask is not None:
            corr = corr * mask
        corr = torch.nn.Softmax(corr, -1) 

        reference_feature = reference_feature.reshape(N, C, H*W).permute(0, 2, 1) 
        reference_features[layer_names[0]] = torch.matmul(corr, reference_feature).reshape(N, H, W, C).permute(0, 3, 1, 2)  
        if other_reference_feature is not None:
            corr_i = corr.reshape(N, 1, H*W, H*W)
            for i, other_reference_feature_i in enumerate(other_reference_feature):
                _, _, feature_h, feature_w = other_reference_feature_i.shape  
                assert H==feature_h*2**(i+1) and W==feature_w*2**(i+1)
                corr_i = self.avgpool(corr_i)
                other_reference_feature_i = other_reference_feature_i.reshape(N, C, H*W//4**(i+1)).permute(0, 2, 1) 
                reference_features[layer_names[i+1]] = torch.matmul(corr_i.squeeze(1), other_reference_feature_i).reshape(N, H//2**(i+1), W//2**(i+1), C).permute(0, 3, 1, 2)

        return reference_features  

class Non_Local_Sparse_Network(nn.Module):
    def __init__( self, N=64, reduction=4, n_hashes=4, chunk_size=144):
        super(Non_Local_Sparse_Network,self).__init__()
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        assert N%self.reduction==0
        self.conv_match = nn.Sequential(ResidualBlock(N, N//reduction),) 
        self.conv_assembly1 = nn.Sequential(ResidualBlock(N, N),) 
        self.conv_assembly2 = nn.Sequential(ResidualBlock(N, N),) 

    def LSH(self, hash_buckets, x):
        #x: [N,2*H*W,C]
        N = x.shape[0]
        device = x.device
        
        #generate random rotation matrix
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets//2) #[1,C,n_hashes,hash_buckets//2]
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1, -1) #[N, C, n_hashes, hash_buckets//2]
        
        #locality sensitive hashing
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations) #[N, n_hashes, 2*H*W, hash_buckets//2]
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1) #[N, n_hashes, 2*H*W, hash_buckets]
        
        #get hash codes
        hash_codes = torch.argmax(rotated_vecs, dim=-1) #[N,n_hashes,2*H*W]
        
        #add offsets to avoid hash codes overlapping between hash rounds 
        offsets = torch.arange(self.n_hashes, device=device) 
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        hash_codes = torch.reshape(hash_codes + offsets, (N, -1,)) #[N,n_hashes*2*H*W]
     
        return hash_codes 
    
    def add_adjacent_buckets(self, x):
            x_extra_back = torch.cat([x[:,:,-1:, ...], x[:,:,:-1, ...]], dim=2)
            x_extra_forward = torch.cat([x[:,:,1:, ...], x[:,:,:1,...]], dim=2)
            return torch.cat([x, x_extra_back,x_extra_forward], dim=3)

    def batched_index_select(self, values, indices):
        last_dim = values.shape[-1]
        return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))         

    def forward(self, feature_dec1, feature_dec2, reference_feature, random_index, is_mask=False, is_detach=True):
        pdb.set_trace()    
        if is_detach:
            feature_dec1 = feature_dec1.detach()
            feature_dec2 = feature_dec2.detach()

        N,C,H,W = feature_dec1.shape
        
        f1_dec_embed = self.conv_match(feature_dec1).view(N,-1,H*W).contiguous().permute(0,2,1) #[N,C,H*W]
        f2_dec_embed = self.conv_match(feature_dec2).view(N,-1,H*W).contiguous().permute(0,2,1) #[N,C,H*W]
        feature_dec1_auxiliary_embed = self.conv_assembly1(feature_dec1).view(N,-1,H*W).contiguous().permute(0,2,1) #[N,H*W,C*reduction]
        reference_feature_embed = self.conv_assembly2(reference_feature).view(N,-1,H*W).contiguous().permute(0,2,1) #[N,H*W,C*reduction]
        L,C = f1_dec_embed.shape[-2:]

        f_device = f1_dec_embed.device
        
        _, inv_random_index = random_index.sort() #[2*H*W]

        #number of hash buckets/hash bits
        hash_buckets = min(2*L//self.chunk_size + (2*L//self.chunk_size)%2, 128)
        
        #get assigned hash codes/bucket number  
        f1f2_dec_embed = torch.concat((f1_dec_embed, f2_dec_embed), 1) #[N,2*H*W,C]      
        hash_codes = self.LSH(hash_buckets, f1f2_dec_embed) #[N,n_hashes*2*H*W]
        hash_codes = hash_codes.detach()
        hash_codes = hash_codes.reshape(N, self.n_hashes, -1).gather(-1, random_index.unsqueeze(0).unsqueeze(0).repeat(N, self.n_hashes, 1)).reshape(N, -1) #[N,n_hashes*2*H*W]

        #group elements with same hash code by sortingcccc
        _, indices = hash_codes.sort(dim=-1) #[N,n_hashes*2*H*W]
        _, undo_sort = indices.sort(dim=-1) #undo_sort to recover original order [N,n_hashes*2*H*W]
        mod_indices = (indices % 2*L) #now range from (0->2*H*W) [N,n_hashes*2*H*W]
        zeros_tensor = torch.zeros_like(f1_dec_embed)[:, :, 0] + 1e-2 #[N,H*W] 
        zeros_ones_tensor = torch.concat((zeros_tensor, 1-zeros_tensor), 1) #[N,2*H*W] 
        f1f2_dec_embed = f1f2_dec_embed.gather(1, random_index.unsqueeze(0).unsqueeze(-1).repeat(N, 1, C)) #[N,2*H*W,C] 
        f1f2_dec_embed_sorted = self.batched_index_select(f1f2_dec_embed, mod_indices) #[N,n_hashes*2*H*W,C]
        zeros_ones_tensor = zeros_ones_tensor.gather(1, random_index.unsqueeze(0).repeat(N, 1))
        zeros_ones_tensor_sorted = zeros_ones_tensor.gather(1, mod_indices) #[N,n_hashes*2*H*W]
        reference_feature_w_f1_embed = torch.concat((feature_dec1_auxiliary_embed, reference_feature_embed), 1) #[N,2*H*W,C*reduction]
        reference_feature_w_f1_embed = reference_feature_w_f1_embed.gather(1, random_index.unsqueeze(0).unsqueeze(-1).repeat(N, 1, C*self.reduction))
        reference_feature_embed_sorted = self.batched_index_select(reference_feature_w_f1_embed, mod_indices) #[N,n_hashes*2*H*W,C*reduction]
        
        #pad the embedding if it cannot be divided by chunk_size
        padding = self.chunk_size - 2*L%self.chunk_size if 2*L%self.chunk_size!=0 else 0
        f1f2_dec_att_buckets = torch.reshape(f1f2_dec_embed_sorted, (N, self.n_hashes,-1, C)) #[N, n_hashes, 2*H*W,C]
        zeros_ones_tensor_att_buckets = torch.reshape(zeros_ones_tensor_sorted, (N, self.n_hashes,-1)) #[N, n_hashes, 2*H*W]
        reference_feature_att_buckets = torch.reshape(reference_feature_embed_sorted, (N, self.n_hashes,-1, C*self.reduction)) #[N, n_hashes, 2*H*W,C*reduction]

        if padding:
            pad_f1f2_dec = f1f2_dec_att_buckets[:,:,-padding:,:].clone()
            pad_zeros_ones_tensor = zeros_ones_tensor_att_buckets[:,:,-padding:].clone()
            pad_reference_feature = reference_feature_att_buckets[:,:,-padding:,:].clone()
            f1f2_dec_att_buckets = torch.cat([f1f2_dec_att_buckets,pad_f1f2_dec],dim=2)
            zeros_ones_tensor_att_buckets = torch.cat([zeros_ones_tensor_att_buckets,pad_zeros_ones_tensor],dim=2)
            reference_feature_att_buckets = torch.cat([reference_feature_att_buckets,pad_reference_feature],dim=2)
        
        f1f2_dec_att_buckets = torch.reshape(f1f2_dec_att_buckets,(N,self.n_hashes,-1,self.chunk_size,C)) #[N, n_hashes, num_chunks, chunk_size, C]
        zeros_ones_tensor_att_buckets = torch.reshape(zeros_ones_tensor_att_buckets,(N,self.n_hashes,-1,self.chunk_size)) #[N, n_hashes, num_chunks, chunk_size]
        reference_feature_att_buckets = torch.reshape(reference_feature_att_buckets,(N,self.n_hashes,-1,self.chunk_size, C*self.reduction)) #[N, n_hashes, num_chunks, chunk_size, C*reduction]
        
        f1f2_dec_match = F.normalize(f1f2_dec_att_buckets, p=2, dim=-1,eps=5e-5) 

        #allow attend to adjacent buckets
        f1f2_dec_match = self.add_adjacent_buckets(f1f2_dec_match) #[N, n_hashes, num_chunks, 3*chunk_size, C]
        zeros_ones_tensor_att_buckets = self.add_adjacent_buckets(zeros_ones_tensor_att_buckets) #[N, n_hashes, num_chunks, 3*chunk_size]
        reference_feature_att_buckets = self.add_adjacent_buckets(reference_feature_att_buckets) #[N, n_hashes, num_chunks, 3*chunk_size, C*reduction]
        
        #unormalized attention score
        raw_score = torch.einsum('bhkie,bhkje->bhkij', f1f2_dec_att_buckets, f1f2_dec_match) #[N, n_hashes, num_chunks, chunk_size, chunk_size*3]
        
        #softmax
        bucket_score = torch.sum(torch.exp(raw_score)*zeros_ones_tensor_att_buckets.unsqueeze(3), dim=-1, keepdim=True) #[N, n_hashes, num_chunks, chunk_size, 1]
        score = torch.exp(raw_score)*zeros_ones_tensor_att_buckets.unsqueeze(3) / bucket_score #(after softmax) [N, n_hashes, num_chunks, chunk_size, chunk_size*3]
        bucket_score = torch.reshape(bucket_score,[N,self.n_hashes,-1]) #[N, n_hashes, 2*H*W+padding]
        
        #attention
        ret = torch.einsum('bukij,bukje->bukie', score, reference_feature_att_buckets) #[N, n_hashes, num_chunks, chunk_size, C*reduction]
        ret = torch.reshape(ret,(N,self.n_hashes,-1,C*self.reduction)) #[N, n_hashes, 2*H*W+padding, C*reduction]
        
        #if padded, then remove extra elements
        if padding:
            ret = ret[:,:,:-padding,:].clone() #[N, n_hashes, 2*H*W, C*reduction]
            bucket_score = bucket_score[:,:,:-padding].clone() #[N, n_hashes, 2*H*W]
         
        #recover the original order
        ret = torch.reshape(ret, (N, -1, C*self.reduction)) #[N, n_hashes*2*H*W,C*reduction]
        bucket_score = torch.reshape(bucket_score, (N, -1,)) #[N,n_hashes*2*H*W]
        ret = self.batched_index_select(ret, undo_sort)#[N, n_hashes*2*H*W,C*reduction]
        bucket_score = bucket_score.gather(1, undo_sort)#[N,n_hashes*2*H*W]
        
        #weighted sum multi-round attention
        ret = torch.reshape(ret, (N, self.n_hashes, 2*L, C*self.reduction)) #[N, n_hashes, 2*H*W,C*reduction]
        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, 2*L, 1)) #[N, n_hashes, 2*H*W,1]
        probs = bucket_score / torch.sum(bucket_score, dim=1, keepdim=True) #[N, n_hashes, 2*H*W,1]
        ret = torch.sum(ret * probs, dim=1) #[N, 2*H*W,C*reduction]
        
        ret = ret.permute(0,2,1).gather(2, inv_random_index.unsqueeze(0).unsqueeze(0).repeat(N, C*self.reduction, 1)).view(N,-1, 2,H,W)[:, :, 0, :, :].contiguous()
        return ret                                 

class Reference_Image_Encoder(nn.Module):

    def __init__(self, N=64, **kwargs):
        super().__init__(**kwargs)
        self.network1 = nn.Sequential(ResidualBlock(3, N),)
        self.network2 = nn.Sequential(ResidualBlockWithStride(N, N, stride=2),
                                      ResidualBlock(N, N),)
        self.network3 = nn.Sequential(ResidualBlockWithStride(N, N, stride=2),
                                      ResidualBlock(N, N),)
        self.network4 = nn.Sequential(ResidualBlockWithStride(N, N, stride=2),
                                      ResidualBlock(N, N),)
        self.network5 = nn.Sequential(ResidualBlockWithStride(N, N, stride=2),
                                      ResidualBlock(N, N),)
                        
    def forward(self, x): 
        f_1 = self.network1(x) 
        f_2 = self.network2(f_1) 
        f_4 = self.network3(f_2)    
        f_8 = self.network4(f_4)
        f_16 = self.network5(f_8)           
        return {"f_1": f_1,"f_2": f_2,"f_4": f_4,"f_8": f_8,"f_16": f_16,}                                             

def create_gaussian_masks(img_h, img_w, patch_h, patch_w):
    #pdb.set_trace()
    """ Creates a set of gaussian maps, each gaussian centered in patch_x center """
    patch_area = patch_h * patch_w
    img_area = img_h * img_w
    num_patches = np.arange(0, img_area // patch_area)
    patch_img_w = img_w / patch_w
    w = np.arange(1, img_w+1, 1, float) - (patch_w % 2)/2
    h = np.arange(1, img_h+1, 1, float) - (patch_h % 2)/2
    h = h[:, np.newaxis]

    # mu = there is a gaussian map centered in each x_patch center
    center_h = (num_patches // patch_img_w + 0.5) * patch_h
    center_w = ((num_patches % patch_img_w) + 0.5) * patch_w

    # gaussian std
    sigma_h = 0.5 * img_h
    sigma_w = 0.5 * img_w

    # create the gaussian maps
    cols_gauss = (w - center_w[:, np.newaxis])[:, np.newaxis, :] ** 2 / sigma_w ** 2
    rows_gauss = np.transpose(h - center_h)[:,:, np.newaxis] ** 2 / sigma_h ** 2
    g = np.exp(-4 * np.log(2) * (rows_gauss + cols_gauss))

    # crop the masks to fit correlation map
    gauss_mask = g[:, (patch_h+1) // 2 - 1:img_h - patch_h // 2,
                (patch_w+1) // 2 - 1:img_w - patch_w // 2]

    return torch.from_numpy(gauss_mask.astype(np.float32)[np.newaxis,:,:,:]).cuda()        


'''
def L2_or_pearson_corr(x, y, patch_h, patch_w):
    """This func calculate the Pearson Correlation Coefficient/L2 between a patch x and all patches in image y
    Formula: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    R =  numerator/ denominator.
    where:
    numerator = sum_i(xi*yi - y_mean*xi - x_mean*yi + x_mean*y_mean)
    denominator = sqrt( sum_i(xi^2 - 2xi*x_mean + x_mean^2)*sum_i(yi^2 - 2yi*y_mean + y_mean^2) )

    Input: tensor of patchs x and img y
    Output: map that each pixel in it, is Pearson correlation/L2 correlative for a patch between x and y
    """
    N, C, H, W = x.shape
    patch_size = int(H * W * C)
    
    weights = nn.Parameter(data=x.data, requires_grad=False)
    xy = F.conv2d(y, weights, padding=0, stride=1)

    kernel_mean = torch.ones(1, C, H, W).cuda()/patch_size
    weights = nn.Parameter(data=kernel_mean.data, requires_grad=False)
    y_mean = F.conv2d(y, weights, padding=0, stride=1)
    x_sum = torch.sum(x, dim = [1, 2 , 3])
    y_mean_x = y_mean * x_sum.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    numerator = xy - y_mean_x

    sum_x_square = torch.sum(torch.square(x), axis=[1, 2, 3])
    x_mean = torch.mean(x, dim = [1, 2 , 3])
    x_mean_x_sum = x_mean*x_sum
    denominator_x = sum_x_square - x_mean_x_sum

    kernel_sum = torch.ones(1, C, H, W).cuda()
    weights = nn.Parameter(data=kernel_sum.data, requires_grad=False)
    sum_y_square = F.conv2d(torch.square(y), weights, padding=0, stride=1)
    y_mean_y_sum = y_mean*y_mean*patch_size
    denominator_y = sum_y_square - y_mean_y_sum

    denominator = denominator_y*denominator_x.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    time.sleep(1e-4)
    del denominator_y, denominator_x, sum_y_square, y_mean_y_sum, sum_x_square, x_mean_x_sum, xy, y_mean_x
    torch.cuda.empty_cache() 
    out = numerator/torch.sqrt(denominator)

    return out
'''
def L2_or_pearson_corr(x, y, patch_h, patch_w, is_cpu=False):
    #pdb.set_trace()
    """This func calculate the Pearson Correlation Coefficient/L2 between a patch x and all patches in image y
    Formula: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    R =  numerator/ denominator.
    where:
    numerator = sum_i(xi*yi - y_mean*xi - x_mean*yi + x_mean*y_mean)
    denominator = sqrt( sum_i(xi^2 - 2xi*x_mean + x_mean^2)*sum_i(yi^2 - 2yi*y_mean + y_mean^2) )

    Input: tensor of patchs x and img y
    Output: map that each pixel in it, is Pearson correlation/L2 correlative for a patch between x and y
    """
    N, C, H, W = x.shape
    patch_size = int(H * W * C)
    
    weights = nn.Parameter(data=x.data, requires_grad=False)
    xy = F.conv2d(y, weights, padding=0, stride=1)

    kernel_mean = torch.ones(1, C, H, W).cuda()/patch_size
    weights = nn.Parameter(data=kernel_mean.data, requires_grad=False)
    y_mean = F.conv2d(y, weights, padding=0, stride=1)
    if is_cpu:
        x = x.cpu()
        y_mean = y_mean.cpu()
        xy = xy.cpu()

    x_sum = torch.sum(x, dim = [1, 2 , 3])
    y_mean_x = y_mean * x_sum.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    numerator = xy - y_mean_x

    sum_x_square = torch.sum(torch.square(x), axis=[1, 2, 3])
    x_mean = torch.mean(x, dim = [1, 2 , 3])
    x_mean_x_sum = x_mean*x_sum
    denominator_x = sum_x_square - x_mean_x_sum

    kernel_sum = torch.ones(1, C, H, W).cuda()
    weights = nn.Parameter(data=kernel_sum.data, requires_grad=False)
    sum_y_square = F.conv2d(torch.square(y), weights, padding=0, stride=1)
    if is_cpu:
        sum_y_square = sum_y_square.cpu()
    y_mean_y_sum = y_mean*y_mean*patch_size
    denominator_y = sum_y_square - y_mean_y_sum
    time.sleep(1e-4)
    del sum_y_square, y_mean_y_sum, sum_x_square, x_mean_x_sum, xy, y_mean_x, y_mean, kernel_sum, weights, x_mean, x_sum, kernel_mean
    torch.cuda.empty_cache() 
    denominator = denominator_y*denominator_x.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    time.sleep(1e-4)
    del denominator_y, denominator_x
    torch.cuda.empty_cache() 
    time.sleep(1e-4)
    torch.cuda.empty_cache() 
    out = numerator/torch.sqrt(denominator)

    if is_cpu:
        return out.cuda()
    else:
        return out 


def reduce_mean_and_std_normalize_images(in_images, ):
    # values from KITTI dataset:
    means = torch.from_numpy(np.array([93.70454143384742, 98.28243432206516, 94.84678088809876])).cuda().float().unsqueeze(0).unsqueeze(2).unsqueeze(2)
    variances = torch.from_numpy(np.array([73.56493292844912, 75.88547006820752, 76.74838442810665])).cuda().float().unsqueeze(0).unsqueeze(2).unsqueeze(2)

    norm_images = (in_images - means)/variances

    return norm_images

def rgb_transform(x, ):
    """This func gets an RGB img tensor (channel last) and transforms it to:
     H1 H2 H3 img
    H1=R+G , H2=R-G , H3= -0.5*(R+B)
    according to: https://pdfs.semanticscholar.org/8120/fa0a8c35e96c7312ab994caa2d47fceb5f85.pdf
    or to LAB"""

    R, G, B = torch.chunk(x, 3, dim=1)
    H1 = R + G
    H2 = R - G
    H3 = 0.5*(R + B)
    x_trans = torch.concat([H1, H2, H3], axis=1)
    return x_trans    


  