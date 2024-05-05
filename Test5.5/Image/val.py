import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset_galaxy import GalaxyClsTinyDataset, create_dataloader
import model_def

from sklearn.metrics import f1_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns  # seaborn库使混淆矩阵更美观

def custom_early_stopping_metric(y_true, y_pred):
    # Calculate F1 scores for class 1 and class 2 separately
    f1_scores = f1_score(y_true, y_pred, labels=[0, 1], average=None)
    f1_class_1 = f1_scores[0]  # F1 score for class 1
    f1_class_2 = f1_scores[1]  # F1 score for class 2
    
    # You can take the average or the minimum value as the early stopping metric
    return (f1_class_1 + f1_class_2) / 2
    # or
    # return min(f1_class_1, f1_class_2)

# Example usage:
# y_true = [1, 2, 1, 2, 1]
# y_pred = [1, 1, 1, 2, 2]
# print(custom_early_stopping_metric(y_true, y_pred))

def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    total_batches = 0
    correct = 0  # 用于累积正确的预测数
    total = 0  # 总样本数
    y_pred = []  # 保存所有预测标签
    y_true = []  # 保存所有真实标签
    global best_loss, epochs_without_improvement
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.cross_entropy(output, target, reduction='sum').item()  # 使用交叉熵损失函数
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()  # 累积正确的预测数
            total += target.size(0)  # 更新总样本数
            y_pred.extend(pred.view_as(target).cpu().numpy())  # 保存预测标签
            y_true.extend(target.cpu().numpy())  # 保存真实标签
            total_batches += 1

    val_loss /= len(val_loader.dataset)
    acc = 100. * correct / total  # 计算准确率
    # 计算F1分数，由于数据不平衡，选择'micro'、'macro'或'weighted'作为平均策略
    f1 = f1_score(y_true, y_pred, average='macro')
    # f1 = custom_early_stopping_metric(y_true, y_pred)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {:.2f}%, F1 Score (Weighted): {:.4f}\n'.format(val_loss, acc, f1))
        
    return y_pred, y_true

def main():
    root_dir = 'data/train_data_10.fits'
    label_dir = 'data/'
    device = "cuda"

    test_dataset = GalaxyClsTinyDataset(data_path=root_dir, label_path=label_dir,
                                        mode='val')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    global best_loss
    # model = model_def.SpecCNN(num_classes=3).to(device) 
    model = model_def.resnet(num_classes=10).to(device) 

    loss_func = nn.CrossEntropyLoss()
    pth_name = 'resnet'
    model.load_state_dict(
        torch.load(f"output/{pth_name}.pth"))

    y_pred, y_true = validate(model, device, test_loader)
    recall = recall_score(y_true, y_pred, average='macro')  # 使用'macro'来计算宏平均recall

    print(f"Recall: {recall}")

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 计算每个类别的准确率
    class_acc = cm.diagonal() / cm.sum(axis=1)

    # 打印每个类别的准确率
    for i, acc in enumerate(class_acc):
        print(f"Class {i} accuracy: {acc*100:.2f}%")

    # 使用seaborn绘制混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    # 添加轴标签
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # 添加标题
    plt.title('Confusion Matrix')

    # 显示图形
    plt.savefig(f"{pth_name}.png")

    


if __name__ == '__main__':
    main()