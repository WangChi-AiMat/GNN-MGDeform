import torch

def pearson_correlation_loss(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)

    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)

    covariance = torch.mean((pred - pred_mean) * (target - target_mean))
    pred_var = torch.var(pred, unbiased=False)
    target_var = torch.var(target, unbiased=False)

    pearson = covariance / (torch.sqrt(pred_var * target_var) + 1e-8)
    return 1 - pearson