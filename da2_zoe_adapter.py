"""
Adapter module to convert DepthAnything-V2 (DINOv2) encoder tokens to 
ZoeDepth-compatible spatial feature pyramid.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add depth_anything_v2 to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_anything_v2.dinov2 import DINOv2
from depth_anything_v2.util.blocks import FeatureFusionBlock, _make_scratch


class TokenToSpatialAdapter(nn.Module):
    """Converts DINOv2 tokens to spatial feature maps at multiple scales."""
    
    def __init__(self, embed_dim=1024, out_channels=[256, 512, 1024, 1024], features=256, use_bn=False):
        """
        Args:
            embed_dim: Embedding dimension from DINOv2 (e.g., 1024 for vitl)
            out_channels: Output channels for each pyramid level
            features: Feature dimension for fusion blocks
            use_bn: Whether to use batch normalization
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        
        # Project tokens to spatial maps
        self.projects = nn.ModuleList([
            nn.Conv2d(embed_dim, out_ch, kernel_size=1, stride=1, padding=0)
            for out_ch in out_channels
        ])
        
        # Resize layers to create pyramid (1/8, 1/16, 1/32, 1/32)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),  # 1/8
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),  # 1/16
            nn.Identity(),  # 1/32 (already at this scale)
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)  # 1/32 (downsample)
        ])
    
    def forward(self, token_features, patch_h, patch_w):
        """
        Args:
            token_features: List of token tensors, each shape (B, N, embed_dim) where N = patch_h * patch_w
            patch_h, patch_w: Spatial dimensions of token grid
        Returns:
            List of spatial feature maps at different scales
        """
        out = []
        for i, tokens in enumerate(token_features):
            # tokens is already patch tokens (from adapter call)
            # Reshape to spatial: (B, N, embed_dim) -> (B, embed_dim, patch_h, patch_w)
            B, N, E = tokens.shape
            
            # Handle case where N might not exactly match (due to padding)
            if N != patch_h * patch_w:
                # Reshape to closest match
                # Assume tokens are in row-major order
                tokens = tokens[:, :patch_h * patch_w, :]
                N = patch_h * patch_w
            
            x = tokens.permute(0, 2, 1).reshape(B, E, patch_h, patch_w)
            
            # Project to target channels
            x = self.projects[i](x)
            
            # Resize to appropriate scale
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        return out


class FeaturePyramidFusion(nn.Module):
    """Fuses pyramid features into a single 1/8-resolution feature map."""
    
    def __init__(self, in_channels=[256, 512, 1024, 1024], features=256, use_bn=False):
        super().__init__()
        
        # Create scratch layers
        self.scratch = _make_scratch(in_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        
        # Feature fusion blocks (FPN-style)
        self.refinenet1 = FeatureFusionBlock(features, nn.ReLU(False), deconv=False, bn=use_bn, align_corners=True)
        self.refinenet2 = FeatureFusionBlock(features, nn.ReLU(False), deconv=False, bn=use_bn, align_corners=True)
        self.refinenet3 = FeatureFusionBlock(features, nn.ReLU(False), deconv=False, bn=use_bn, align_corners=True)
        self.refinenet4 = FeatureFusionBlock(features, nn.ReLU(False), deconv=False, bn=use_bn, align_corners=True)
        
        # Final bottleneck projection
        self.bottleneck = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
    
    def forward(self, pyramid_features):
        """
        Args:
            pyramid_features: List of feature maps [1/8, 1/16, 1/32, 1/32]
        Returns:
            Fused feature map at 1/8 resolution
        """
        layer_1, layer_2, layer_3, layer_4 = pyramid_features
        
        # Project all to same feature dimension
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        # Bottom-up fusion
        path_4 = self.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.refinenet1(path_2, layer_1_rn)
        
        # Final bottleneck
        bottleneck = self.bottleneck(path_1)
        
        return bottleneck, [layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn]


class DepthAnythingV2ZoeCore(nn.Module):
    """
    Core module: DepthAnything-V2 encoder + adapter + feature fusion.
    Outputs features compatible with ZoeDepth metric head.
    """
    
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], 
                 use_bn=False, use_clstoken=False):
        super().__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder_name = encoder
        self.pretrained_encoder = DINOv2(model_name=encoder)
        embed_dim = self.pretrained_encoder.embed_dim
        
        # Token to spatial adapter
        self.adapter = TokenToSpatialAdapter(
            embed_dim=embed_dim,
            out_channels=out_channels,
            features=features,
            use_bn=use_bn
        )
        
        # Feature pyramid fusion
        self.fusion = FeaturePyramidFusion(
            in_channels=out_channels,
            features=features,
            use_bn=use_bn
        )
        
        # Store output channels for ZoeDepth compatibility
        self.output_channels = [features] + out_channels  # [bottleneck, layer1, layer2, layer3, layer4]
    
    def forward(self, x, return_rel_depth=False, denorm=False):
        """
        Args:
            x: Input image tensor (B, 3, H, W)
            return_rel_depth: Whether to return relative depth (for compatibility)
            denorm: Whether input is denormalized (unused, kept for API compatibility)
        Returns:
            tuple: (relative_depth, [bottleneck, layer1, layer2, layer3, layer4])
        """
        B, C, H, W = x.shape
        patch_h = H // 14
        patch_w = W // 14
        
        # Get intermediate features from encoder
        # Returns list of (patch_tokens, cls_token) tuples if return_class_token=True
        features = self.pretrained_encoder.get_intermediate_layers(
            x, 
            self.intermediate_layer_idx[self.encoder_name],
            reshape=False,
            return_class_token=True
        )
        
        # Extract patch tokens from tuple format (same as DPTHead)
        # features is list of tuples: [(patch_tokens, cls_token), ...]
        patch_features = [feat[0] for feat in features]  # Extract patch tokens
        
        # Convert tokens to spatial feature maps
        pyramid_features = self.adapter(patch_features, patch_h, patch_w)
        
        # Fuse pyramid into single feature map
        bottleneck, multi_scale_features = self.fusion(pyramid_features)
        
        # For compatibility with ZoeDepth, return relative depth placeholder
        # (DepthAnything encoder produces relative features, not explicit depth)
        rel_depth = None
        if return_rel_depth:
            # Create a relative depth signal from bottleneck features
            # Use a simple projection to get depth-like signal
            # This acts as a relative depth cue for the metric head
            rel_depth_feat = torch.mean(bottleneck, dim=1, keepdim=True)  # (B, 1, H, W)
            rel_depth = F.interpolate(
                rel_depth_feat, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=True
            ).squeeze(1)  # (B, H, W)
        
        # Return in ZoeDepth format: [bottleneck, layer1, layer2, layer3, layer4]
        output = [bottleneck] + multi_scale_features
        
        if return_rel_depth:
            return rel_depth, output
        return output

