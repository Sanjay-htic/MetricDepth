import torch
 
def eval_depth(pred, target):
    """
    Compute depth evaluation metrics.
    
    Args:
        pred: Predicted depth tensor, shape (..., N) where N is number of pixels
        target: Ground truth depth tensor, same shape as pred
        
    Returns:
        Dictionary of metrics
    """
    assert pred.shape == target.shape
    
    # Ensure values are positive (should already be clamped, but safety check)
    eps = 1e-6
    pred = torch.clamp(pred, min=eps)
    target = torch.clamp(target, min=eps)

    # Compute threshold metrics (delta metrics)
    thresh = torch.max((target / pred), (pred / target))
    
    # Use numel() to get total number of elements, not len() which gives first dimension
    num_elements = thresh.numel()
    
    # Delta metrics: percentage of pixels within threshold
    d1 = torch.sum(thresh < 1.25).float() / num_elements
    d2 = torch.sum(thresh < 1.25 ** 2).float() / num_elements
    d3 = torch.sum(thresh < 1.25 ** 3).float() / num_elements

    # Compute differences
    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)
    
    # Relative errors
    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    # RMSE metrics
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

    # Log10 error
    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    
    # Scale invariant log error
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {
        'd1': d1.item(), 
        'd2': d2.item(), 
        'd3': d3.item(), 
        'abs_rel': abs_rel.item(), 
        'sq_rel': sq_rel.item(), 
        'rmse': rmse.item(), 
        'rmse_log': rmse_log.item(), 
        'log10': log10.item(), 
        'silog': silog.item()
    }