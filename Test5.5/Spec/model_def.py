import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecCNNComplex(nn.Module):
    def __init__(self, num_classes):
        super(SpecCNNComplex, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # 第二个卷积块
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # 第三个卷积块
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # 新增第四个卷积块
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(num_features=256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # 全连接层
        # 注意：这里的输入特征维度需要根据实际输入数据的维度来调整
        self.fc1 = nn.Linear(2816, 512)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)  # 扁平化
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class SpecCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpecCNN, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(num_features=16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # 第二个卷积块
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # 第三个卷积块
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # 全连接层
        # self.fc1 = nn.Linear(64 * 93, 512)  # 假设经过卷积和池化后，特征长度为93
        self.fc = nn.Linear(2944, num_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # 扁平化
        x = self.drop(x)
        x = self.fc(x)
        return x
    
class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()
        # 1x1 卷积分支
        self.branch1x1 = nn.Conv1d(in_channels, 24, kernel_size=1)

        # 1x1 卷积后接 3x3 卷积分支
        self.branch3x3_1 = nn.Conv1d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv1d(16, 24, kernel_size=3, padding=1)

        # 1x1 卷积后接 5x5 卷积分支
        self.branch5x5_1 = nn.Conv1d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv1d(16, 24, kernel_size=5, padding=2)

        # 3x3 池化后接 1x1 卷积分支
        self.branch_pool_1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_2 = nn.Conv1d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)

        # 将四个分支的结果在深度上拼接
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class SpecNet(nn.Module):
    def __init__(self, num_classes):
        super(SpecNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 应用两个Inception模块
        self.inception1 = InceptionModule(in_channels=10)
        self.inception2 = InceptionModule(in_channels=96)  # 4个分支，每个分支输出24个通道

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(96, num_classes)  # 与inception2的输出通道数一致

    def forward(self, x):
        # 确保输入是 [batch_size, 1, 3000]
        if x.dim() == 2:
            x = x.unsqueeze(1)  
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # 假设我们有10个类别
    # model = SpecNet(num_classes=3)
    # model = SpecCNN(num_classes=3)
    model = SpecCNNComplex(num_classes=3)
    # 假设输入是 [4, 3000] 的张量
    input_tensor = torch.ones([4,3000])
    print(model)