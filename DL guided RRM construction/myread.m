clear;clc
filename = 'C:\Users\HP\PycharmProjects\untitled\try11.xlsx';
network_parameters = {};
sheets = {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'Extra_Params'};
for i = 1:length(sheets)
    sheet_name = sheets{i};
    if strcmp(sheet_name, 'Extra_Params')
        extra_params = readtable(filename, 'Sheet', sheet_name);
    else
        matrix = readmatrix(filename, 'Sheet', sheet_name);
        network_parameters{i} = matrix; % 存储到 cell 数组中
    end
end
fc1_weight = network_parameters{1};
fc1_bias = network_parameters{2};
fc2_weight = network_parameters{3};
fc2_bias = network_parameters{4};
