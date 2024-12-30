clear;clc
load("data.mat")
for i=1:24
    temp=data{i};
    data{i}=[mean(temp(:,1:5),2),temp(:,6)];
end
% PA=[69.4664903647806,60.8518646691342,81.0743731243526,65.8869789138279,80.4612347850364,119.963220629360,108.788818185976,119.969249389715,97.1948313490155,106.240350713313,97.7412327179733,62.8425329883416,41.0824490950951,77.7537277860686,44.9594270833178,15.1401295359264,106.150661322574,74.9355532851193,85.5245092311990,76.8107835344512,35.9750243221304,59.4725093348607,81.3118594913570,67.5061511144474];
PA=[69.4664903647806,60.8518646691342,81.0743731243526,65.8869789138279,80.4612347850364,119.963220629360,108.788818185976,119.969249389715,97.1948313490155,106.240350713313,97.7412327179733,62.8425329883416,41.0824490950951,77.7537277860686,44.9594270833178,15.1401295359264,106.150661322574,74.9355532851193,85.5245092311990,76.8107835344512,35.9750243221304,59.4725093348607,81.3118594913570,67.5061511144474];
canshu(1,1:4)=[0,50,nan,14];%D
canshu(2,1:4)=[0,5,nan,0];%EP
canshu(3,1:4)=[90,250,nan,125];%WM
canshu(4,1:4)=[0.1,0.4,nan,0.4];%B
canshu(5,1:4)=[0,10,nan,8.9];%FC
canshu(6,1:4)=[0.5,0.9999,nan,0.91];%CG
canshu(7,1:4)=[0.01,10,nan,10];%N
canshu(8,1:4)=[0.01,10,nan,0.177];%K
canshu(9,1:4)=[0,100,nan,15];%新U
canshu(10,1:4)=[0,10,nan,0.5];%新V

% 遗传算法参数
nvars = size(canshu,1);
lb = canshu(:,1)';
ub = canshu(:,2)';
global bb
bb = canshu(:,4)';

% 适应度函数
fitnessFcn = @(para) fl(data,para,PA);

% 完善设置
options = optimoptions('ga', 'PopulationSize', 100, ...
    'MaxGenerations', 5000, ...
    'Display', 'iter', ...
    'OutputFcn', @outfun, ...
    'FunctionTolerance', 1e-6); % 设置FunctionTolerance为0就是无限迭代

% 运行遗传算法
[optimal_para, optimal_loss] = hgd(fitnessFcn, nvars, [], [], [], [], lb, ub, [], options);
save('temp.mat');

% 输出结果
disp(['最优参数：', num2str(optimal_para)]);
disp(['最小损失：', num2str(optimal_loss)]);

% 画图
para=optimal_para;
for i=1:24
    subplot(4,6,i)
    P=data{i}(:,1);
    qobs=data{i}(:,2);
    qsim=basicmodel(P, qobs,para,PA(i));
    doc3(i)=fnse(qobs,qsim);
    doc1(i)=(max(qsim)-max(qobs))/max(qobs);
    doc2(i)=(sum(qsim)-sum(qobs))/sum(qobs);
    yyaxis left;
    xlabel('时段 (1h)')
    ylabel('流量 (m^3/s)')
    hold on;
    plot(qobs, 'LineWidth', 2);
    plot(qsim);
    ylim([0 max(max(qsim),max(qobs))/0.7]);
    yyaxis right;
    ylabel('降雨 (mm)')
    ba=bar(P);
    set(ba,'EdgeColor','None');
    ylim([0 max(P) / 0.25]);
    set(gca, 'Ydir', 'reverse');
    legend('实测流量', '模拟流量', '降雨');
    box on;
end

number=4;
load('doc.mat')
doc{number,1}=doc1;
doc{number,2}=doc2;
doc{number,3}=doc3;
save doc.mat doc

% 自定义输出函数，可通过检测 stop.txt 文件来优雅地中断 GA
function [state, options, optchanged] = outfun(options, state, flag)
optchanged = false; % 默认没有改变选项
if exist('stop.txt', 'file') % 检查文件是否存在
    disp('Stop file detected. Stopping GA...');
    state.StopFlag = 'y'; % 设置停止标志
    optchanged = true; % 标记为需要改变选项，以停止算法
end
end


% 损失函数定义
function loss = fl(data,para,PA)
doc=zeros(1,24);
for i=1:24
    P=data{i}(:,1);
    qobs=data{i}(:,2);
    qsim=model3(P, qobs,para,PA(i));
    doc(i)=fnse(qobs,qsim);
end
loss=mean(doc);
end