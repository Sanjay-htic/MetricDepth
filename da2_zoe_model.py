"""
Combined model: DepthAnything-V2 encoder + ZoeDepth metric head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from da2_zoe_adapter import DepthAnythingV2ZoeCore
import sys
import os

# Add ZoeDepth to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ZoeDepth'))
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.layers.attractor import AttractorLayer, AttractorLayerUnnormed
from zoedepth.models.layers.dist_layers import ConditionalLogBinomial
from zoedepth.models.layers.localbins_layers import Projector, SeedBinRegressor, SeedBinRegressorUnnormed


class DepthAnythingV2ZoeDepth(DepthModel):
    """
    Combined model: DepthAnything-V2 encoder + ZoeDepth metric head.
    Uses pretrained DepthAnything-V2 encoder for relative depth features,
    then applies ZoeDepth's metric bin head for metric depth prediction.
    """
    
    def __init__(
        self,
        encoder='vitl',
        n_bins=64,
        bin_centers_type="softplus",
        bin_embedding_dim=128,
        min_depth=1e-3,
        max_depth=10,
        n_attractors=[16, 8, 4, 1],
        attractor_alpha=300,
        attractor_gamma=2,
        attractor_kind='sum',
        attractor_type='inv',  # Use inverse attractor (better stability)
        min_temp=5,
        max_temp=50,
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        train_encoder=True,
        encoder_lr_factor=10,
        **kwargs
    ):
        """
        Args:
            encoder: DINOv2 encoder variant ('vits', 'vitb', 'vitl', 'vitg')
            n_bins: Number of depth bins
            bin_centers_type: 'normed', 'softplus', 'hybrid1', 'hybrid2'
            bin_embedding_dim: Dimension for bin embeddings
            min_depth, max_depth: Depth range
            n_attractors: List of attractor counts at each refinement level
            attractor_alpha: Attractor strength parameter
            attractor_gamma: Attractor exponential parameter
            attractor_kind: 'sum' or 'mean' for aggregation
            attractor_type: 'inv' (inverse) or 'exp' (exponential)
            min_temp, max_temp: Temperature range for log-binomial distribution
            features: Feature dimension for fusion
            out_channels: Output channels for pyramid levels
            use_bn: Whether to use batch normalization
            train_encoder: Whether to train the encoder
            encoder_lr_factor: Learning rate factor for encoder
        """
        super().__init__()
        
        self.core = DepthAnythingV2ZoeCore(
            encoder=encoder,
            features=features,
            out_channels=out_channels,
            use_bn=use_bn
        )
        
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.min_temp = min_temp
        self.bin_centers_type = bin_centers_type
        self.train_encoder = train_encoder
        self.encoder_lr_factor = encoder_lr_factor
        
        btlnck_features = self.core.output_channels[0]
        num_out_features = self.core.output_channels[1:]
        
        # Bottleneck convolution
        self.conv2 = nn.Conv2d(btlnck_features, btlnck_features, kernel_size=1, stride=1, padding=0)
        
        # Choose bin regressor and attractor types
        if bin_centers_type == "normed":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayer
        elif bin_centers_type == "softplus":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid1":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid2":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayer
        else:
            raise ValueError("bin_centers_type should be one of 'normed', 'softplus', 'hybrid1', 'hybrid2'")
        
        # Seed bin regressor (initial metric hypothesis)
        self.seed_bin_regressor = SeedBinRegressorLayer(
            btlnck_features, 
            n_bins=n_bins, 
            min_depth=min_depth, 
            max_depth=max_depth
        )
        
        # Projectors for bin embeddings
        self.seed_projector = Projector(btlnck_features, bin_embedding_dim)
        self.projectors = nn.ModuleList([
            Projector(num_out, bin_embedding_dim)
            for num_out in num_out_features
        ])
        
        # Attractors for bin refinement
        self.attractors = nn.ModuleList([
            Attractor(
                bin_embedding_dim, 
                n_bins, 
                n_attractors=n_attractors[i],
                min_depth=min_depth, 
                max_depth=max_depth,
                alpha=attractor_alpha, 
                gamma=attractor_gamma, 
                kind=attractor_kind, 
                attractor_type=attractor_type
            )
            for i in range(len(num_out_features))
        ])
        
        # Conditional log-binomial distribution head
        # Input: bottleneck features + relative depth placeholder (we'll use features directly)
        N_FEATURES = features + 1  # +1 for relative depth placeholder
        self.conditional_log_binomial = ConditionalLogBinomial(
            N_FEATURES, 
            bin_embedding_dim, 
            n_classes=n_bins, 
            min_temp=min_temp, 
            max_temp=max_temp
        )
    
    def forward(self, x, return_final_centers=False, denorm=False, return_probs=False, **kwargs):
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            return_final_centers: Whether to return final bin centers
            denorm: Whether input is denormalized (unused, kept for API)
            return_probs: Whether to return probability distribution
        
        Returns:
            dict with 'metric_depth' key, and optionally 'bin_centers', 'probs'
        """
        B, C, H, W = x.shape
        
        # Get features from core (DepthAnything encoder + adapter + fusion)
        rel_depth, out = self.core(x, return_rel_depth=True, denorm=denorm)
        
        # Extract bottleneck and multi-scale features
        outconv_activation = out[0]  # bottleneck at 1/8 resolution
        btlnck = out[0]
        x_blocks = out[1:]  # [layer1, layer2, layer3, layer4]
        
        # Process bottleneck
        x_d0 = self.conv2(btlnck)
        x = x_d0
        
        # Seed bin regressor (initial metric hypothesis)
        _, seed_b_centers = self.seed_bin_regressor(x)
        
        # Normalize bin centers if needed
        if self.bin_centers_type == 'normed' or self.bin_centers_type == 'hybrid2':
            b_prev = (seed_b_centers - self.min_depth) / (self.max_depth - self.min_depth)
        else:
            b_prev = seed_b_centers
        
        # Initial bin embedding
        prev_b_embedding = self.seed_projector(x)
        
        # Refine bins through attractor layers
        for projector, attractor, x_block in zip(self.projectors, self.attractors, x_blocks):
            b_embedding = projector(x_block)
            b, b_centers = attractor(
                b_embedding, 
                b_prev, 
                prev_b_embedding, 
                interpolate=True
            )
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()
        
        # Prepare for log-binomial distribution
        # Use bottleneck features + relative depth signal
        last = outconv_activation
        
        # Create relative depth condition from features (normalized)
        # Interpolate rel_depth to match bottleneck size
        rel_cond = rel_depth.unsqueeze(1)  # (B, 1, H, W)
        rel_cond = F.interpolate(rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)
        
        # Normalize relative depth per-sample to [0, 1] range for stability
        B = rel_cond.shape[0]
        rel_cond_normalized = torch.zeros_like(rel_cond)
        for b in range(B):
            rel_b = rel_cond[b]
            rel_min, rel_max = rel_b.min(), rel_b.max()
            if rel_max > rel_min:
                rel_cond_normalized[b] = (rel_b - rel_min) / (rel_max - rel_min)
            else:
                rel_cond_normalized[b] = rel_b
        
        # Concatenate with bottleneck features
        last = torch.cat([last, rel_cond_normalized], dim=1)
        
        # Interpolate bin embedding to match feature size
        b_embedding = F.interpolate(
            b_embedding, 
            last.shape[-2:], 
            mode='bilinear', 
            align_corners=True
        )
        
        # Compute probability distribution over bins
        x_probs = self.conditional_log_binomial(last, b_embedding)
        
        # Interpolate bin centers to match probability size
        b_centers = F.interpolate(
            b_centers, 
            x_probs.shape[-2:], 
            mode='bilinear', 
            align_corners=True
        )
        
        # Compute final depth: expected value of bins
        out_depth = torch.sum(x_probs * b_centers, dim=1, keepdim=True)
        
        # Structure output
        output = dict(metric_depth=out_depth)
        
        if return_final_centers or return_probs:
            output['bin_centers'] = b_centers
        
        if return_probs:
            output['probs'] = x_probs
        
        return output
    
    def get_lr_params(self, lr):
        """
        Get learning rate parameters for different components.
        
        Args:
            lr: Base learning rate
        
        Returns:
            List of parameter groups with different learning rates
        """
        param_conf = []
        
        if self.train_encoder:
            # Encoder parameters
            encoder_params = list(self.core.pretrained_encoder.parameters())
            param_conf.append({
                'params': encoder_params,
                'lr': lr / self.encoder_lr_factor
            })
        
        # Adapter and fusion parameters
        adapter_params = list(self.core.adapter.parameters())
        fusion_params = list(self.core.fusion.parameters())
        param_conf.append({
            'params': adapter_params + fusion_params,
            'lr': lr
        })
        
        # Metric head parameters (bins, attractors, etc.)
        remaining_modules = []
        for name, child in self.named_children():
            if name != 'core':
                remaining_modules.append(child)
        
        remaining_params = []
        for child in remaining_modules:
            remaining_params.extend(list(child.parameters()))
        
        param_conf.append({
            'params': remaining_params,
            'lr': lr
        })
        
        return param_conf

