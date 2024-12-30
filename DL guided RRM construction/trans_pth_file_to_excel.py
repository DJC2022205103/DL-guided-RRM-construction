import torch
import pandas as pd
import numpy as np
from openpyxl import Workbook

# 加载 .pth 文件
for i in range(1,2):
    # path1 = fr"C:\Users\HP\PycharmProjects\untitled\try13-2({i}).pth"
    path1 = r"C:\Users\HP\PycharmProjects\untitled\trytemp.pth"
    state_dict = torch.load(path1)

    # 创建一个新的 Excel 文件
    with pd.ExcelWriter(path1.replace('.pth', '.xlsx'), engine='openpyxl') as writer:
        # 存储所有的网络参数
        for key, value in state_dict.items():
            if 'fc' in key:  # 处理网络参数（例如 fc1, fc2）
                matrix = value.numpy()
                df = pd.DataFrame(matrix)
                df.to_excel(writer, sheet_name=key, index=False, header=False)

        # 存储额外的水文模型参数
        extra_params = {k: v.numpy().flatten() for k, v in state_dict.items() if 'theta' in k}
        extra_params_df = pd.DataFrame(extra_params)
        extra_params_df.to_excel(writer, sheet_name='Extra_Params', index=False)

    print("Conversion to try12.xlsx completed.")
