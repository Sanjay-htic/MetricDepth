"""
Simple test script to verify the model can be instantiated and run forward pass.
"""
import torch
import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ZoeDepth'))

from da2_zoe_model import DepthAnythingV2ZoeDepth

def test_model():
    """Test model instantiation and forward pass."""
    print("Testing DepthAnything-V2 + ZoeDepth model...")
    
    # Create model
    print("Creating model...")
    model = DepthAnythingV2ZoeDepth(
        encoder='vitl',
        n_bins=64,
        bin_centers_type='softplus',
        min_depth=0.1,
        max_depth=10.0,
        train_encoder=False  # Don't train encoder for test
    )
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dummy input
    batch_size = 2
    height, width = 384, 384
    x = torch.randn(batch_size, 3, height, width)
    
    print(f"\nRunning forward pass with input shape: {x.shape}...")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output['metric_depth'].shape}")
    print(f"Output depth range: [{output['metric_depth'].min():.3f}, {output['metric_depth'].max():.3f}]")
    
    print("\n✅ Model test passed!")
    return True

if __name__ == '__main__':
    try:
        test_model()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
