import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import csv

import model_def

from sklearn.metrics import f1_score
from astropy.io import fits
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns  # seaborn库使混淆矩阵更美观

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

class SpecClsDataset(Dataset):

    def __init__(self,
                 data_path='./', 
                 mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hdulist = fits.open(data_path)
        self.mode = mode

    def __len__(self):
        return len(self.hdulist[0].data)

    def __getitem__(self, idx):
        flux = self.hdulist[0].data[idx]
        objid = self.hdulist[1].data['objid'][idx]
        flux = torch.tensor(spec_norm(flux)).float()
        return flux, objid

def validate(model, device, val_loader):
    model.eval()
    y_pred = []  # 保存所有预测标签
    id_list = []
    with torch.no_grad():
        for data, id in val_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测结果
            y_pred.extend(pred[0].cpu().numpy())  # 保存预测标签
            id_list.append(id.item())

    return y_pred, id_list

def main():
    root_dir = 'data/test_data.fits'
    label_dir = 'data/'
    device = "cuda"

    test_dataset = SpecClsDataset(data_path=root_dir, 
                                        mode='val')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    global best_loss
    # model = model_def.SpecCNN(num_classes=3).to(device) 
    model = model_def.SpecCNNComplex(num_classes=3).to(device) 

    pth_name = 'specnet'
    model.load_state_dict(
        torch.load(f"output/{pth_name}.pth"))

    y_pred, id_list = validate(model, device, test_loader)


    # 创建CSV文件并写入数据
    with open('prediction_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['#objid', 'label'])
        # 写入数据
        for objid, label in zip(id_list, y_pred):
            writer.writerow([objid, label])
    

    


if __name__ == '__main__':
    main()