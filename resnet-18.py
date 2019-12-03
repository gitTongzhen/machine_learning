from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torchvision import models
# 设置batch_size
batch_size = 16
# 设置学习率
learning_rate = 0.001
# 设置epoch
epoch = 10

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),   #随机大小，随机长宽比裁剪原始图片，最后将图片resize到设定好的size
    transforms.RandomHorizontalFlip(),   #依据概率p对PIL图片进行水平翻转，默认的p是0.5
    transforms.ToTensor(),               #转化为tensor 归一化到0-1之间，
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))  #数据标准化，先减去0.5，再除以标准差
])
val_transforms = transforms.Compose([
    transforms.Resize(256),             #resize图像到256*256
    transforms.RandomResizedCrop(224),  #随机大小，随机长宽比裁剪原始图片，最后将图片resize到设定好的size
    transforms.ToTensor(),              #转化为tensor 归一化到0-1之间，
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])


# train_dir = 'D:\\Desktop\\workspace\\github\\machine_learning\\machine_learning\\VGGDataSet\\train'
train_dir = '/docker_gpu/tf_mtcnn/test/VGGDataSet/train'
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
print(type(train_datasets))
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

# val_dir = 'D:\\Desktop\\workspace\\github\\machine_learning\\machine_learning\\VGGDataSet\\val'
val_dir = '/docker_gpu/tf_mtcnn/test/VGGDataSet/val'
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
print(len(val_datasets.imgs))
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True)


class VGGNet(nn.Module):
    def __init__(self, num_classes=2):
        super(VGGNet, self).__init__()
        # 加载vgg16的模型参数和网络结构
        net = models.resnet18(pretrained=True)
        # 设定网络的分类器
        net.classifier = nn.Sequential()
        # 网络的特征层
        self.features = net
        # 网络的分类器自己重新设定好
        self.classifier = nn.Sequential(
                # 线性变换层的参数设置
                nn.Linear(1000, 512),
                # inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
                nn.ReLU(True),
                # 随机丢弃一些特征值
                nn.Dropout(),

                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#--------------------训练过程---------------------------------
model = VGGNet()
if torch.cuda.is_available():
    model.cuda()
params = [{'params': md.parameters()} for md in model.children()
          if md in [model.classifier]]
# adam优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# 交叉熵损失函数
loss_func = nn.CrossEntropyLoss()

Loss_list = []
Accuracy_list = []
from torchsummary import summary
# net = models.resnet18(pretrained=True)
# summary(net, (3, 224, 224))


for epoch in range(100):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_dataloader:
        batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
        # 第一步，将数据导入到模型中
        out = model(batch_x)
        #第二步，利用交叉熵计算代价
        loss = loss_func(out, batch_y)

        train_loss +=  loss.item()
        pred = torch.max(out, 1)[1]
        # 计算正确率
        train_correct = (pred == batch_y).sum()
        #print("1111111",train_correct)
        train_acc += train_correct.item()
        #将模型的参数梯度初始化为0
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 更新所有的参数
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_datasets)), train_acc / (len(train_datasets))))
    torch.save(model,'vgg.pth')
    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in val_dataloader:
        batch_x, batch_y = Variable(batch_x, volatile=True).cuda(), Variable(batch_y, volatile=True).cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        # 统计acc的准确率
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
        #需要打印出loss和acc的准确率
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        val_datasets)), eval_acc / (len(val_datasets))))
    
    Loss_list.append(eval_loss / (len(val_datasets)))
    Accuracy_list.append(100 * eval_acc / (len(val_datasets)))

x1 = range(0, 100)
x2 = range(0, 100)
result = dict()
result["ACC"] = Accuracy_list
result["loss"] = Loss_list
with open("test.json") as f:
    json.dump(result,f)
y1 = Accuracy_list
y2 = Loss_list

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()
plt.savefig("accuracy_loss.jpg")
