def mae(pred, target):
    return (pred.view_as(target) - target).abs().mean().item()
