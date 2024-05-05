from astropy.io import fits
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



def create_dataloader(path, label, batch_size):
    train_dataset = SpecClsDataset(data_path=path, label_path=label,
                                        mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    val_dataset = SpecClsDataset(data_path=path, label_path=label,
                                        mode='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class SpecClsDataset(Dataset):

    def __init__(self,
                 data_path='./', label_path = './',
                 mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hdulist = fits.open(data_path)
        with open(label_path+mode+'.txt', 'r') as f:
            self.label_list = f.readlines()
        self.mode = mode

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        idx_t = eval(self.label_list[idx])
        flux = self.hdulist[0].data[idx_t]
        objid = self.hdulist[1].data['objid'][idx_t]
        label = torch.tensor(self.hdulist[1].data['label'][idx_t])
        flux = torch.tensor(spec_norm(flux)).float()
        return flux, label



# Preprocessing
def spec_norm(fluxes_original):
    # 找到非零值的最小值和最大值
    flux_min = fluxes_original.min()
    flux_max = fluxes_original.max()

    # 进行归一化处理
    fluxes_normalized = (fluxes_original - flux_min) / (flux_max - flux_min)
    # 注意：此方法会在fluxes_interpolated中的零值处产生负值，因为它们现在低于flux_min
    # 如果你想在归一化后的数据中保持这些零值不变，可以将它们重新设置为0
    fluxes_normalized[fluxes_original == 0] = 0

    return fluxes_normalized

def split_train_val():
    import random

    total_data = 100000
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

    hdulist = fits.open('data/test_data.fits')
    label = hdulist[1].data
    label = hdulist[0].data
    print(len(label))
    # from collections import Counter
    # # 元素 0 出现的次数为：3427
    # # 元素 1 出现的次数为：1011
    # # 元素 2 出现的次数为：95562

    # second_elements = [element[1] for element in label]
    # element_counts = Counter(second_elements)
    # for element, count in element_counts.items():
    #     print(f"元素 {element} 出现的次数为：{count}")
    dataset = SpecClsDataset('data/train_data_10.fits', 'data/', mode='train')
    idx = 69000
    flux, label = dataset.__getitem__(idx)
    wavelength = np.linspace(3900,9000,3000)
    print(flux.dtype)#type(torch.tensor(flux, dtype=float)))

    # c = {0:'GALAXY',1:'QSO',2:'STAR'}
    # plt.plot(wavelength,flux)
    # plt.title(f'class:{c[label]}')
    # plt.xlabel('wavelength ({})'.format(f'$\AA$'))
    # plt.ylabel('flux')
    # plt.show()
