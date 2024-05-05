import argparse
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset_spec import SpecClsDataset, create_dataloader
import model_def

best_loss = 0#float('inf')
epochs_without_improvement = 0
from sklearn.metrics import f1_score
import torch.nn.functional as F

def custom_early_stopping_metric(y_true, y_pred):
    # Calculate F1 scores for class 1 and class 2 separately
    f1_scores = f1_score(y_true, y_pred, labels=[0, 1], average=None)
    f1_class_1 = f1_scores[0]  # F1 score for class 1
    f1_class_2 = f1_scores[1]  # F1 score for class 2
    
    # You can take the average or the minimum value as the early stopping metric
    return (f1_class_1 + f1_class_2) / 2

def validate(args, model, device, val_loader, patience=5):
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
    f1 = custom_early_stopping_metric(y_true, y_pred)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {:.2f}%, Score (Weighted): {:.4f}\n'.format(val_loss, acc, f1))
    
    # 使用F1分数作为提前停止的指标
    if f1 > best_loss:  # 注意这里改为比较F1分数
        best_loss = f1
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print('Early stopping! No improvement in F1 Score for {} epochs.'.format(patience))
        return False
    
    return f1  # 或者返回f1分数，取决于您后续如何使用这个函数的返回值

# def validate(args, model, device, val_loader, patience=5):
#     model.eval()
#     criterion = nn.L1Loss()#StdLoss(mode='val')
#     val_loss = 0
#     correct = 0
#     total_batches = 0
#     global best_loss, epochs_without_improvement
    
#     with torch.no_grad():
#         for data, target in val_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
            
#             val_loss += criterion(output, target).item()#.item()#nn.L1Loss()(output, target).item()  # sum up batch loss
#             total_batches += 1

#     val_loss /= total_batches
#     print('\nValidation set: Average loss (MAE): {:.4f}\n'.format(val_loss))
    
#     if val_loss < best_loss:
#         best_loss = val_loss
#         epochs_without_improvement = 0
#     else:
#         epochs_without_improvement += 1

#     if epochs_without_improvement >= patience:
#         print('Early stopping! No improvement in validation loss for {} epochs.'.format(patience))
#         return False
    
#     return val_loss



def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    # criterion = nn.CrossEntropyLoss()#$StdLoss(mode='train')
    for batch_idx, data in enumerate(train_loader):
        flux, label = data
        flux = flux.to(device)
        label= label.to(device)

        optimizer.zero_grad()
        output = model(flux)
        # print(output.shape)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        # alpha 可以是一个浮点数或者浮点数列表，对应每个类别的权重
        if alpha is None:
            self.alpha = torch.tensor(1.0)
        else:
            self.alpha = torch.tensor(alpha, device='cuda')
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算softmax概率
        probs = F.softmax(inputs, dim=1)
        # 将targets转化为one-hot编码
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        # 计算每个类别的focal loss
        focal_loss = -self.alpha * ((1.0 - probs) ** self.gamma) * (targets_one_hot * torch.log(probs + 1e-8))
        
        # 根据reduction参数决定输出
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--early_stop', type=int, default=8, metavar='N',
                        help='number of epochs to stop')
    parser.add_argument('--save_name', type=str, default='new_model', metavar='N',
                        help='the name of your model to save')
    parser.add_argument('--model', type=str, default='simple', metavar='N',
                        help='the model you choose')
    parser.add_argument('--loss', type=str, default='simple', metavar='N',
                        help='the loss you choose')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--augment', action='store_true', default=False)
    
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    root_dir = 'data/train_data_10.fits'
    label_dir = 'data/'
    device = "cuda"

    train_loader, val_loader = create_dataloader(root_dir, label_dir, args.batch_size)

    global best_loss
    if args.model == 'specnet':
        model = model_def.SpecCNNComplex(num_classes=3).to(device)  
    elif args.model == 'simple':
        model = model_def.SpecCNN(num_classes=3).to(device) 

    if args.loss == 'focal':
        # 类别的相对频率
        frequencies = [95.5, 3.5, 1]
        alpha_inverse = [1.0 / f for f in frequencies]
        alpha_sum = sum(alpha_inverse)
        alpha_normalized = [a / alpha_sum for a in alpha_inverse]
        loss_func = FocalLoss(alpha=alpha_normalized, gamma=2.0)
    elif args.loss == 'simple':
        class_counts = [3.5, 1, 95.5]
        inverse_weights = [1.0 / count for count in class_counts]
        normalized_weights = [weight / sum(inverse_weights) * len(class_counts) for weight in inverse_weights]
        weights_tensor = torch.tensor(normalized_weights, dtype=torch.float, device='cuda')
        loss_func = nn.CrossEntropyLoss(weight=weights_tensor)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    train_losses = []
    val_losses = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch, loss_func)
        val_loss = validate(args, model, device, val_loader, patience=args.early_stop)
        
        # scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # print(best_loss)
        # early stop
        # val_loss = val_acc
        if val_loss == False:
            break
        # if val_loss <= best_loss:
        if val_loss >= best_loss:
            print('save_losses:', val_loss)
            torch.save(model.state_dict(), f"output/{args.model}.pth")
        

    # Plotting the training and validation loss
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses,label="train")
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(f"output/{args.model}_losses.png")

    with open(f"output/{args.model}_losses.txt", 'w') as f:
        for num, i in enumerate(val_losses):
            f.write(str(num) + '\t' + str(i) + '\n') 

if __name__ == '__main__':
    main()
    
