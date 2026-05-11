import os
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torchvision.transforms as transforms

class OfflineMNIST(TorchDataset):
    '''
    该类是 管家类
    主要负责读取和处理数据集 
    transform是 代码后续自定义的transform方法(处理图片像素值的方法)
    '''
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        # 若是训练模式 则读取训练集数据
        if train:
            image_path = os.path.join(root, "MNIST", "raw", "train-images-idx3-ubyte.gz")
            label_path = os.path.join(root, "MNIST", "raw", "train-labels-idx1-ubyte.gz")
        else:
            image_path = os.path.join(root, "MNIST", "raw", "t10k-images-idx3-ubyte.gz")
            label_path = os.path.join(root, "MNIST", "raw", "t10k-labels-idx1-ubyte.gz")

        self.images = self.read_images(image_path)
        self.labels = self.read_labels(label_path)

    def read_images(self, path):# 读取图片 每张图片从第17个字节处开始读
        with gzip.open(path, "rb") as f:
            f.read(16)
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            return arr.reshape(-1, 28, 28)

    def read_labels(self, path):# 读取标签 每个标签从第9个字节处开始读
        with gzip.open(path, "rb") as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)
    
    # 重载len和get方法
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

        if self.transform:
            img = self.transform(img)

        return img, label


def get_mnist_loaders(data_dir, batch_size=64):
    '''
    自定义了一个数据加载器方法 使用管家类里的数据加载方式
    '''
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = OfflineMNIST(root=data_dir, train=True, transform=transform)
    test_dataset = OfflineMNIST(root=data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader