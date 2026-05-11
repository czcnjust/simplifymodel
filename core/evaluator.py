import os
import torch

def evaluate_model(model, test_loader, device):
    '''
    定义评估模型性能的方法
    '''
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_file_size_mb(path):
    if not os.path.exists(path):
        return 0
    return round(os.path.getsize(path) / 1024 / 1024, 4)