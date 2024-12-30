%【函数】nash系数的计算
function nse =fnse(Qobs, Qsim)
    Qobs=Qobs(:);Qsim=Qsim(:);
    minl=min(length(Qobs),length(Qsim));
    Qobs=Qobs(1:minl);
    Qsim=Qsim(1:minl);
    % 计算观测值的平均值
    obs_mean = mean(Qobs);
    
    % 计算Nash-Sutcliffe效率系数
    nse = 1 - sum((Qobs - Qsim).^2) / sum((Qobs - obs_mean).^2);
end