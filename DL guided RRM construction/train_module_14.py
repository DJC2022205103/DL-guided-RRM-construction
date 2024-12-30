#684+1
import math
import torch.nn.functional as Fun
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

# 读取excel文件
def readexcel(path):
    excel_file = pd.ExcelFile(path)
    data_dict = {}
    data_dict2 = {}
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name, header=None)
        data_array = df.to_numpy()
        data_dict[sheet_name] = data_array[:, 0:2]
        data_dict2[sheet_name] = data_array
    third_columns = [data_dict2[key][0:10, 2] for key in sorted(data_dict2.keys(), key=int)]
    initial_value = np.array(third_columns)
    return data_dict, initial_value

# 把数据转换成x,y和loader
def prepare_data(data, initial_value, split):
    span = max(len(data[str(i + 1)]) for i in range(len(data)))
    x = []
    y = []
    initials = []
    for i in range(len(data)):
        temp = data[str(i + 1)]
        if len(temp) == span:
            _x = temp[:, 0]
            _y = temp[:, 1]
        else:
            _x = np.concatenate([np.zeros([span - len(temp)]), temp[:, 0]])
            _y = np.concatenate([np.zeros([span - len(temp)]), temp[:, 1]])
        x.append(_x)
        y.append(_y)
        initials.append(initial_value[i])  # 添加初值

    x = np.array(x)
    y = np.array(y)
    initials = np.array(initials)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)  # Add an extra dimension for input_size
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    initials = torch.tensor(initials, dtype=torch.float32)  # 转换为张量

    # 确保split的长度为3，并且split的总和等于x的第一个维度
    assert len(split) == 3
    assert sum(split) == x.shape[0]

    x_splits = torch.split(x, split)
    y_splits = torch.split(y, split)
    initials_splits = torch.split(initials, split)

    val_loader = DataLoader(TensorDataset(x_splits[0], y_splits[0], initials_splits[0]), batch_size=32) if split[0] != 0 else None
    train_loader = DataLoader(TensorDataset(x_splits[1], y_splits[1], initials_splits[1]), batch_size=32, shuffle=True) if split[1] != 0 else None
    test_loader = DataLoader(TensorDataset(x_splits[2], y_splits[2], initials_splits[2]), batch_size=32) if split[2] != 0 else None

    return val_loader, train_loader, test_loader

# 水文模型
class HydrologicalModel(pl.LightningModule):
    def __init__(self, s1):
        super(HydrologicalModel, self).__init__()
        self.fc1 = nn.Linear(s1[0][0], s1[0][1])
        self.fc2 = nn.Linear(s1[0][1], s1[0][2])
        self.fc3 = nn.Linear(s1[1][0], s1[1][1])
        self.fc4 = nn.Linear(s1[1][1], s1[1][2])
        self.fc5 = nn.Linear(s1[2][0], s1[2][1])
        self.fc6 = nn.Linear(s1[2][1], s1[2][2])
        self.criterion = nn.MSELoss()

    def forward(self, x, initial_values):
        outputs = torch.tensor([], requires_grad=True)
        state1 = torch.zeros_like(x[:, 0, :])
        state2 = torch.zeros_like(x[:, 0, :])
        for t in range(x.size(1)):
            p = x[:, t, :]
            in1 = p
            temp1 = self.fc2(torch.relu(self.fc1(in1)))  # (batch_size, output_size)
            out1 = temp1[:, 0].unsqueeze(1)
            in2 = torch.cat((out1, state1), dim=1)
            temp2 = self.fc4(torch.relu(self.fc3(in2)))  # (batch_size, output_size)
            out2 = temp2[:, 0].unsqueeze(1)
            state1 = temp2[:, 1].unsqueeze(1)
            in3 = torch.cat((out2, state2), dim=1)
            temp3 = self.fc6(torch.relu(self.fc5(in3)))  # (batch_size, output_size)
            out3 = temp3[:, 0].unsqueeze(1)
            state2 = temp3[:, 1].unsqueeze(1)
            outputs = torch.cat((outputs,out3), dim=1)
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, targets, initial_values = batch
        outputs = self(inputs, initial_values)
        outputs = torch.where(torch.isnan(outputs), torch.full_like(outputs, 0), outputs)
        loss = self.criterion(outputs.unsqueeze(-1), targets)
        self.log('train_loss', loss)
        print(f'Train Loss: {loss.item()}')  # 添加这一行来打印loss值
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

# 数据准备
data, initial_value = readexcel('C:\\Users\\HP\\PycharmProjects\\untitled\\洪水.xlsx')  # 从excel表读取数据
val_loader, train_loader, test_loader = prepare_data(data, initial_value, [0, len(data), 0])  # 处理数据为dataloader

# 实例化并训练模型
model = HydrologicalModel(
    s1=[[1,3,1],[2,3,2],[2,3,2]]
)

trainer = pl.Trainer(max_epochs=10000)
trainer.fit(model, train_loader)