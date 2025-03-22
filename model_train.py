from torchvision.datasets import FashionMNIST
from torchvision import transforms  # 处理数据集的
import torch.utils.data as Data
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from model import VGG16
import matplotlib.pyplot as plt
import copy  # 导入 copy 模块
import time  # 导入 time 模块

# 处理数据集和验证集
def train_val_data_process():
    # 加载数据，模型输入为1*224*224
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                              download=True
                              )
    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                      batch_size=32,
                                      shuffle=True,
                                      num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=2)
    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义优化器，b梯度下降法（Adam优化更新参数，移动权重平均值，抑制梯度消失或爆炸，加速梯度下降）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 损失函数,多分类用交叉熵损失，回归用均方差损失
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    # 复制模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 最高准确度,初始化
    best_acc = 0.0
    # 训练集loss列表
    train_loss_all = []
    # 验证集loss列表
    val_loss_all = []
    # 训练集acc列表
    train_acc_all = []
    # 验证集acc列表
    val_acc_all = []
    # 保存时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # 初始化参数，训练集损失和准确度，验证集损失和准确度
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0

        # 初始化参数，训练集和验证集数量
        train_num = 0
        val_num = 0

        # 一批次一批次处理数据
        for _, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征和标签输入设备
            b_x = b_x.to(device)  # 128*1*28*28
            b_y = b_y.to(device)  # 128*label

            # 打开训练模式
            model.train()

            output = model(b_x)

            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)

            loss = criterion(output, b_y)

            # 将梯度初始化为0，防止前面的梯度累加
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 参数更新
            optimizer.step()
            # 对每一轮损失进行累加,loss是每个batch每个样本的平均值
            train_loss += loss.item() * b_x.size(0)

            # 对每一轮准确度进行累加，若预测正确，sum+=1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 累加训练集个数
            train_num += b_x.size(0)
        for _, (b_x, b_y) in enumerate(val_dataloader):
            # 将验证特征和标签输入设备
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置为评估模型
            model.eval()
            # 前向传播计算结果
            output = model(b_x)
            # 计算正确的类别
            pre_lab = torch.argmax(output, dim=1)
            # 计算验证损失
            loss = criterion(output, b_y)
            # 对每一轮验证损失进行累加
            val_loss += loss.item() * b_x.size(0)
            # 对每一轮验证准确度进行累加，若预测正确，sum+=1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 累加个数
            val_num += b_x.size(0)
        # 计算保留每一轮Epoch结果
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        # 计算训练耗时
        time_use = time.time() - since
        print("The use time of Train and Valid: {:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                          "train_loss_all": train_loss_all,
                                          "val_loss_all": val_loss_all,
                                          "train_acc_all": train_acc_all,
                                          "val_acc_all": val_acc_all
                                          })
    # 选择最优的参数
    # 加载最高准确率下的模型参数
    torch.save(best_model_wts, './checkpoints/best_model.pth')
    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, 'ro-', label='train loss')
    plt.plot(train_process['epoch'], train_process.val_loss_all, 'bs-', label='val loss')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, 'ro-', label='train acc')
    plt.plot(train_process['epoch'], train_process.val_acc_all, 'bs-', label='val acc')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig('Loss_and_acc.png')


if __name__ == "__main__":
    # 加载模型
    LeNet = VGG16()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet, train_dataloader, val_dataloader, 30)
    matplot_acc_loss(train_process)