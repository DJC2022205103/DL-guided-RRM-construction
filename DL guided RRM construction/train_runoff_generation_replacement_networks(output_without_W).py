#与try12类似，也是产流网络，2-5-1，这次W不出网络，循环计算
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
import os

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
    def __init__(self, input_size, hidden_size, output_size, theta, theta_min, theta_max):
        super(HydrologicalModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.theta = nn.ParameterList([nn.Parameter(torch.tensor(param)) for param in theta])
        self.theta_min = torch.tensor(theta_min, dtype=torch.float32)
        self.theta_max = torch.tensor(theta_max, dtype=torch.float32)
        self.criterion = nn.MSELoss()

    def forward(self, x, initial_values):
        input_size = self.fc1.in_features
        hidden_size = self.fc1.out_features
        output_size = self.fc2.out_features
        outputs = torch.tensor([], requires_grad=True)
        W = initial_values[:, 8].unsqueeze(1)
        QG0 = initial_values[:, 9].unsqueeze(1)
        F = initial_values[:, 1].unsqueeze(1)
        dt = initial_values[:, 0].unsqueeze(1)*10
        WM = self.theta[2]
        qt0 = QG0
        qsim0 = QG0
        rs_doc = torch.empty(F.size()[0], 0)
        rg_doc = torch.empty(F.size()[0], 0)
        sp0 = torch.zeros_like(W)
        N = self.theta[6]
        K = self.theta[7]
        T = x.shape[1]
        #以下生成单位线uh
        t = torch.arange(0, T, device=x.device, dtype=x.dtype)  # 生成 t=1,2,...,T-1
        uh = torch.zeros(T, device=x.device, dtype=x.dtype)  # 初始化uh
        gamma_N = torch.exp(torch.lgamma(N))  # 替代 torch.special.gamma(N)
        uh[0:] = 1 / (K * gamma_N) * ((t+1) / K) ** (N - 1) * torch.exp(-(t+1) / K)  # 矢量化的公式
        #主循环迭代
        for t in range(x.size(1)):
            p = x[:, t, :]
            pe, sp0 = self.evapotranspiration(p, sp0)
            input = torch.cat((pe, W), dim=1)
            temp = self.fc2(torch.relu(self.fc1(input)))  # (batch_size, output_size)
            r = torch.relu(temp[:, 0].unsqueeze(1))
            W = pe - r + W
            rs, rg = self.sourcesdividing(pe, r)
            rs_doc = torch.cat((rs_doc, rs), dim=1)
            rg_doc = torch.cat((rg_doc, rg), dim=1)
        #汇流在主循环以外
        qsim = self.conflow(rs_doc, rg_doc, F, dt, QG0, uh)
        outputs = qsim
        return outputs

    def evapotranspiration(self, p, sp0):
        D = self.theta[0]
        EP = self.theta[1]
        pe = torch.zeros_like(p)
        sp = sp0 + p
        for i in range(len(p)):
            if sp0[i] >= D:
                pe[i] = p[i]
            elif sp[i] <= D:
                pe[i] = 0
            else:
                pe[i] = sp[i] - D
            pe[i] = max(0, pe[i] - EP)
        return pe, sp

    def runoffgeneration(self, pe, W):
        WM = self.theta[2]
        B = self.theta[3]
        WMM = (1 + B) * WM
        r = torch.zeros_like(pe)
        W2 = torch.zeros_like(W)
        for i in range(len(pe)):
            if pe[i] > 0:
                if WM - W[i] <= 0:
                    A = WMM
                else:
                    A = WMM * (1 - (1 - W[i] / WM) ** (1 / (1 + B)))
                if pe[i] + A < WMM:
                    r[i] = (pe[i] - WM + W[i] + WM * (1 - (pe[i] + A) / WMM) ** (1 + B))
                else:
                    r[i] = pe[i] - WM + W[i]
            else:
                r[i] = 0
            W2[i] = W[i] + pe[i] - r[i]
        return r, W2

    def sourcesdividing0(self, pe, r):
        FC = self.theta[4]
        rs = torch.zeros_like(pe)
        rg = torch.zeros_like(pe)
        for i in range(len(r)):
            if pe[i] != 0:
                rg[i] = r[i] * min(pe[i], FC) / pe[i]
                rs[i] = r[i] - rg[i]
        return rs, rg

    def sourcesdividing(self, pe, r):
        FC = 10
        rs = torch.zeros_like(pe)
        rg = torch.zeros_like(pe)
        for i in range(len(r)):
            if pe[i] == 0:
                temp = 0.01
            else:
                temp = pe[i]
            rg[i] = r[i] * min(temp, FC) / temp
            rs[i] = r[i] - rg[i]
        return rs, rg

    def conflow(self, rs_doc, rg_doc, F, dt, QG0, uh):
        CG = self.theta[5]
        QT = torch.zeros_like(rs_doc)
        for i in range(rs_doc.shape[0]):
            rsi = rs_doc[i,:]
            rgi = rg_doc[i,:]
            QG0i = QG0[i]
            QGi = torch.zeros_like(rsi) # Initialize the output tensor
            for t in range(rs_doc.shape[1]):
                QGi[t] = QG0i * CG + rgi[t] * (1 - CG) * F[0] / 3.6 / dt[0]
                QG0i = QGi[t].clone()
            QSi = self.convolve_vectors(rsi, uh) * F[0] / 3.6 / dt[0]
            QT[i, :] = QSi + QGi
        return QT

    def convolve_vectors(self, rsi, uh):
        rsi_length = rsi.size(0)  # rsi 的长度
        uh_length = uh.size(0)  # uh 的长度
        conv_result = torch.zeros(rsi_length)
        for i in range(rsi_length):
            uh_part = uh[:i + 1]  # 取 uh 的前 i+1 个元素
            uh_part_reversed = uh_part.flip(0)  # 反转取到的 uh 部分
            conv_result[i] = torch.sum(rsi[max(0, i - uh_length + 1):i + 1] * uh_part_reversed)
        return conv_result

    def training_step(self, batch, batch_idx):
        inputs, targets, initial_values = batch
        outputs = self(inputs, initial_values)
        outputs = torch.where(torch.isnan(outputs), torch.full_like(outputs, 0), outputs)
        loss = self.criterion(outputs.unsqueeze(-1), targets)
        self.log('train_loss', loss)
        print(f'Train Loss: {loss.item()}')  # 添加这一行来打印loss值
        return loss

    def on_after_backward(self):
        for i, param in enumerate(self.theta):
            param.data = torch.clamp(param.data, self.theta_min[i], self.theta_max[i])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def print_hydrological_params(self):
        print("Evapotranspiration Parameters:")
        print(f"D:{self.theta[0].item()}")
        print(f"EP:{self.theta[1].item()}")
        print("Runoff-generation Parameters:")
        print(f"WM:{self.theta[2].item()}")
        print(f"B:{self.theta[3].item()}")
        print("Sources-dividing Parameters:")
        print(f"FC:{self.theta[4].item()}")
        print("Conflow Parameters:")
        print(f"CG:{self.theta[5].item()}")
        print("Routing Parameters:")
        print(f"N:{self.theta[6].item()}")
        print(f"K:{self.theta[7].item()}")

# 启用异常检测
torch.autograd.set_detect_anomaly(True)

# 数据准备
data, initial_value = readexcel('C:\\Users\\HP\\PycharmProjects\\untitled\\洪水.xlsx')  # 从excel表读取数据
val_loader, train_loader, test_loader = prepare_data(data, initial_value, [0, len(data), 0])  # 处理数据为dataloader

# 示例参数，可以根据具体需求进行调整
canshu = np.zeros((8, 4))
canshu[0,0:4]=[0,50,6,0.4813]#D
canshu[1,0:4]=[0,5,6,1.6437]#EP
canshu[2,0:4]=[90,250,6,178.9307]#WM
canshu[3,0:4]=[0.1,0.4,6,0.3917]#B
canshu[4,0:4]=[0,10,6,5.1647]#FC
canshu[5,0:4]=[0.01,0.99,6,0.9039]#CG
canshu[6,0:4]=[0.01,10,6,3.6825]#N
canshu[7,0:4]=[0.01,10,6,0.6058]#K
theta = canshu[:, 3].tolist()
theta_min=canshu[:,0].tolist()
theta_max=canshu[:,1].tolist()

# # 实例化并训练模型
# model = HydrologicalModel(
#     input_size=2,
#     hidden_size=5,
#     output_size=2,
#     theta=theta,
#     theta_min=theta_min,
#     theta_max=theta_max
# )

# 画洪水
def plotflood(train_loader, model):
    col = 5
    row = math.ceil(24 / col)  # 确保画出24个子图
    model.eval()
    with torch.no_grad():
        for i in range(len(train_loader.dataset)):  # 遍历所有24个数据点
            inputs, targets, initial_values = train_loader.dataset[i]
            inputs = inputs.unsqueeze(0)  # 增加批次维度
            targets = targets.unsqueeze(0)  # 增加批次维度
            initial_values = initial_values.unsqueeze(0)  # 增加批次维度

            qsim = model(inputs, initial_values)
            seq_len = inputs.size(1)  # 获取序列长度
            x = np.arange(seq_len)

            targets_np = targets.squeeze().numpy()
            qsim_np = qsim.squeeze().numpy()
            inputs_np = inputs.squeeze().numpy()

            ax1 = plt.subplot(row, col, i + 1)
            # 实测流量
            color = 'tab:blue'
            ax1.scatter(x, targets_np, s=10, color=color)  # s是点的大小
            ax1.tick_params(axis='y', labelcolor=color)
            # 预报流量
            ax1.plot(x, qsim_np, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_ylim(0, max(max(qsim_np), max(targets_np)) / 0.7)
            # 雨量
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.bar(x, inputs_np.flatten(), color=color, alpha=0.6)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(max(inputs_np.flatten()) / 0.25, 0)
        plt.show()

# trainer = pl.Trainer(max_epochs=1000)
# trainer.fit(model, train_loader)
# plotflood(train_loader, model)
# torch.save(model.state_dict(), 'try15.pth')

# # 打印水文模型参数
# model.print_hydrological_params()

# # 修改主逻辑，加入循环
# for i in range(1, 101):
#     # 实例化并训练模型
#     model = HydrologicalModel(
#         input_size=2,
#         hidden_size=5,
#         output_size=1,
#         theta=theta,
#         theta_min=theta_min,
#         theta_max=theta_max
#     )
#
#     # 创建文件名
#     output_filename = f'try16({i}).pth'
#
#     # 实例化训练器并训练模型
#     trainer = pl.Trainer(
#         max_epochs=1000,
#         logger=False,  # 禁用日志记录
#         enable_checkpointing=False  # 禁用检查点保存
#     )
#     trainer.fit(model, train_loader)
#
#     # 保存模型状态字典
#     torch.save(model.state_dict(), output_filename)
#
#     print(f'Model saved as {output_filename}')
#
# # 打印水文模型参数
#     model.print_hydrological_params()




# 初始化存储损失的字典
losses = {}

# 生成所有 .pth 文件的路径
base_path = r'C:\Users\HP\PycharmProjects\untitled\产流结果储存'  # 替换为实际路径
pth_files = [os.path.join(base_path, f'try16({i}).pth') for i in range(1, 74)]  # 修改范围以适应你的文件数量

# 遍历所有 .pth 文件
for pth_file in pth_files:
    # 实例化模型
    model = HydrologicalModel(
        input_size=2,
        hidden_size=5,
        output_size=1,
        theta=theta,
        theta_min=theta_min,
        theta_max=theta_max
    )

    # 加载模型参数
    model.load_state_dict(torch.load(pth_file))
    model.eval()  # 设置模型为评估模式

    # 计算损失
    total_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():  # 不需要计算梯度
        for batch in train_loader:
            inputs, targets, initial_values = batch
            outputs = model(inputs, initial_values)
            outputs = torch.where(torch.isnan(outputs), torch.full_like(outputs, 0), outputs)  # 处理 NaN 值
            loss = criterion(outputs.unsqueeze(-1), targets)
            total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    losses[pth_file] = average_loss
    print(f'Loss for {pth_file}: {average_loss}')

# 打印所有损失
print("\nAll Losses:")
for pth_file, loss in losses.items():
    print(f"{pth_file}: {loss}")