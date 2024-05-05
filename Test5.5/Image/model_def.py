import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Effnet(nn.Module):
    def __init__(self, num_classes=10):
        super(Effnet, self).__init__()
        self.num_cls = num_classes
        
        self.backbone = models.efficientnet_v2_s(pretrain=True)
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        hidden_layer_features = 128
        # self.head = nn.Sequential(
        #     nn.Linear(512, hidden_layer_features),
        #     nn.ReLU(),
        #     nn.Dropout(0.5), # Dropout率设为0.5，可以根据需要调整
        #     nn.Linear(hidden_layer_features, 2) # 假设最终分类的类别数为2
        #     )
        self.head = nn.Sequential(
            nn.Linear(1280, 128), # 假设最终分类的类别数为2
            nn.Dropout(0.4), # Dropout率设为0.5，可以根据需要调整
            nn.Linear(128, num_classes)
            # nn.ReLU()
            )
        # self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        # x = self.softmax(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks=[2, 2, 2, 2], num_classes=10):# [3, 4, 6, 3]
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Dropout(0.4), # Dropout率设为0.5，可以根据需要调整
            nn.Linear(512, num_classes)
            # nn.ReLU()
            )

        # self.heatmap_conv = nn.Conv2d(512, 1, kernel_size=1)
        # self.dsnt = dsntnn.dsnt

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        x= self.avgpool(out)
        x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        x = self.head(x)

        return x
    
def resnet(num_classes=10):
    model = ResNet(BasicBlock, num_classes=num_classes)
    return model

if __name__ == '__main__':
    # 假设我们有10个类别
    # model = SpecNet(num_classes=3)
    # model = SpecCNN(num_classes=3)
    model = resnet(num_classes=10)
    # 假设输入是 [4, 3000] 的张量
    input_tensor = torch.ones([4,3,224,224])
    print(model)