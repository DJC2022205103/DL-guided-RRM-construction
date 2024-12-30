function [optimal_para, optimal_loss] = hgd(fitnessFcn, nvars, ~, ~, ~, ~, lb, ub, ~, options)
% population_size = options.PopulationSize;
max_generations = options.MaxGenerations;
tolerance = options.FunctionTolerance;


H=fitnessFcn;
theta_min=lb;
theta_max=ub;
theta=rand(1,nvars).*(theta_max-theta_min)+theta_min;
global PA
theta(end-24+1:end)=PA;


%步骤一：定义HGD算法输入
N=max_generations;                %迭代次数，5000次差不多
u=0.05;                 %Z与Zbest的相对偏差theta,state)
n1=1000;                 %收敛判定条件
n2=2000;                %放宽的收敛判定条件
s=0.01;                 %步长系数
k0=90;                  %放大系数
v=100;                  %模式切换间隔
w=1.1;                  %步长变化因子

%步骤二：HGD初始化
i=1;
Z=H(theta);
Z_best=Z;
theta_best=theta;
D=length(theta);
alpha=ones(D,1);
Z0=-inf;
n=0;

while 1
    %步骤三：确定搜索轮次
    if (n>n1 && Z_best-Z>u*Z_best) || (n>n2 && Z_best-Z<=u*Z_best)% || imag(alpha(l))~=0 || isnan(alpha(l)) || abs(alpha(l))==inf
        theta=theta_best;
        Pp=1-log(i)/log(N);
        for d=1:D
            if rand<Pp
                theta(d)=rand*(theta_max(d)-theta_min(d))+theta_min(d);
            end
        end
        Z=H(theta);
        alpha=ones(D,1);
        Z0=-inf;
        n=0;
    end
    %步骤四：确定搜索模式
    if mod(floor(i/v),2)==1
        k=rand*k0*s*(theta_max-theta_min);
    else
        k=alpha*s.*(theta_max-theta_min);
    end
    k(15)=1;
    %步骤五：计算伪梯度并调整参数
    for d=1:D
        theta_origin=theta;
        theta_add=theta;
        theta_minus=theta;
        theta_add(d)=theta_add(d)+k(d);
        theta_add(theta_add<theta_min)=theta_min(theta_add<theta_min);
        theta_add(theta_add>theta_max)=theta_max(theta_add>theta_max);
        Z_add=H(theta_add);
        theta_minus(d)=theta_minus(d)-k(d);
        theta_minus(theta_minus<theta_min)=theta_min(theta_minus<theta_min);
        theta_minus(theta_minus>theta_max)=theta_max(theta_minus>theta_max);
        Z_minus=H(theta_minus);
        if Z_add>Z_minus && Z_add>Z
            Z=Z_add;
            theta=theta_add;
            alpha(d)=alpha(d)*w;
        elseif Z_minus>Z && Z_minus>Z_add
            Z=Z_minus;
            theta=theta_minus;
            alpha(d)=alpha(d)*w;
        else
            alpha(d)=alpha(d)/w;
        end
    end
    %步骤六：记录和更新最优参数
    if Z==Z0
        n=n+1;
    end
    Z0=Z;
    if Z>Z_best
        Z_best=Z;
        theta_best=theta;
    end
    %步骤七：重复步骤三至六
    disp([i/N Z_best]);
    if i>N
        break;
    else
        i=i+1;
    end
end
optimal_para=theta_best;
optimal_loss=Z_best;