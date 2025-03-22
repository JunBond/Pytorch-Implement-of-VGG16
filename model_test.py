from torchvision.datasets import FashionMNIST
from torchvision import transforms  # 处理数据集的
import torch.utils.data as Data
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from model import VGG16

# 处理测试集
def test_data_process():
    # 加载数据，模型输入为1*224*224
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                              download=True
                              )
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0)
    return test_dataloader

def test_model_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #初始化模型参数
    test_corrects = 0.0
    test_num = 0

    with torch.no_grad(): #约定俗成的写法, 因为没有梯度反向传播计算
        for test_x, test_y in test_dataloader: #因为batchsize为1，一张一张推理
            # 将验证特征和标签输入设备
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            # 设置为评估模型
            model.eval()
            # 前向传播计算结果
            output = model(test_x)
            # 计算正确的类别
            pre_lab = torch.argmax(output, dim=1) #沿着第一维度找最大值
            test_corrects += torch.sum(pre_lab == test_y.data)
            # 累加个数
            test_num += test_x.size(0)
    test_acc = test_corrects.double().item() / test_num
    print("Test Acc:", test_acc)


def infer_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1) #沿着第一维度找最大值
            result = pre_lab.item() #获取数值
            label = b_y.item()
            print("预测值：",result,"-------","真实值",label)

if __name__ == "__main__":
    #load model
    model = VGG16()
    model.load_state_dict(torch.load('./checkpoints/best_model.pth'))

    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)

