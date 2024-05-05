from astropy.io import fits
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import h5py
from torchvision import transforms
from PIL import Image


def create_dataloader(path, label, batch_size):
    train_dataset = GalaxyClsDataset(data_path=path, label_path=label,
                                        mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    val_dataset = GalaxyClsDataset(data_path=path, label_path=label,
                                        mode='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def create_dataloader_tiny(path, label, batch_size):
    train_dataset = GalaxyClsTinyDataset(data_path=path, label_path=label,
                                        mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    val_dataset = GalaxyClsTinyDataset(data_path=path, label_path=label,
                                        mode='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



class GalaxyClsDataset(Dataset):

    def __init__(self, data_path='./',
                 label_path = './',
                 mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with h5py.File('data/Galaxy10.h5', 'r') as F:
            images = np.array(F['images'])
            labels = np.array(F['ans'])

        # {0: 3461, 1: 6997, 2: 6292, 3: 349, 4: 1534, 5: 17, 6: 589, 7: 1121, 8: 906, 9: 519}
        # unique, counts = np.unique(labels, return_counts=True)
        # class_counts = dict(zip(unique, counts))
        # print(class_counts)

        with open(label_path+mode+'.txt', 'r') as f:
            self.sample_list = f.readlines()
        # To convert to desirable type
        self.labels = labels.astype(np.float32)
        self.images = images.astype(np.float32)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 将图像缩放到224x224
            transforms.ToTensor(),  # 将PIL图像转换为张量
            transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
            transforms.RandomRotation(20)  # 随机旋转图像±20度
        ])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        idx_t = eval(self.sample_list[idx])
        image = self.images[idx_t]
        pil_image = Image.fromarray((image).astype(np.uint8))
        label = self.labels[idx_t]
        label = torch.tensor(label, dtype=torch.long)
        transformed_image = self.transform(pil_image)
        return transformed_image, label

class GalaxyClsTinyDataset(Dataset):

    def __init__(self, data_path='./',
                 label_path = './',
                 mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with h5py.File('data/Galaxy10.h5', 'r') as F:
            images = np.array(F['images'])
            labels = np.array(F['ans'])

        # {0: 3461, 1: 6997, 2: 6292, 3: 349, 4: 1534, 5: 17, 6: 589, 7: 1121, 8: 906, 9: 519}
        # unique, counts = np.unique(labels, return_counts=True)
        # class_counts = dict(zip(unique, counts))
        # print(class_counts)

        with open(label_path+mode+'.txt', 'r') as f:
            self.sample_list = f.readlines()
        # To convert to desirable type
        self.labels = labels.astype(np.float32)
        self.images = images.astype(np.float32)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将PIL图像转换为张量
            transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
            transforms.RandomRotation(20)  # 随机旋转图像±20度
        ])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        idx_t = eval(self.sample_list[idx])
        image = self.images[idx_t]
        pil_image = Image.fromarray((image).astype(np.uint8))
        label = self.labels[idx_t]
        label = torch.tensor(label, dtype=torch.long)
        transformed_image = self.transform(pil_image)
        return transformed_image, label

def split_train_val():
    import random

    total_data = 21785
    # 设置训练集占比
    train_ratio = 0.7

    # 生成总数据序号列表
    data_indices = list(range(total_data))
    # 随机采样得到训练集序号列表
    train_indices = random.sample(data_indices, int(total_data * train_ratio))
    # 对训练集序号进行排序
    train_indices.sort()

    # 得到验证集序号列表
    val_indices = list(set(data_indices) - set(train_indices))
    # 对验证集序号进行排序
    val_indices.sort()

    # 将训练集序号写入train.txt
    with open('data/train.txt', 'w') as f:
        for index in train_indices:
            f.write(f"{index}\n")

    # 将验证集序号写入val.txt
    with open('data/val.txt', 'w') as f:
        for index in val_indices:
            f.write(f"{index}\n")

    print("文件写入完成。")


if __name__ == '__main__':
    # split_train_val()

    dataset = GalaxyClsDataset(label_path='data/', mode='train')
    img, label = dataset.__getitem__(5)
    print(label)

    