import torch
import torch.nn as nn
import torch.optim as optim

def check_tensor_validity(tensor, name):
    if torch.isnan(tensor).any():
        print(f" NaN ！")
        return False
    if torch.isinf(tensor).any():
        print(f"Inf ！")
        return False
    return True

def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm