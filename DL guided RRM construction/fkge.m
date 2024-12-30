%【函数】kge系数的计算
function kg = fkge(Qobs, Qsim)
    Qobs=Qobs(:);
    Qsim=Qsim(:);
    % 计算观测值和模拟值的平均值
    obs_mean = mean(Qobs);
    sim_mean = mean(Qsim);

    % 计算Kling-Gupta系数
    numerator = sum((Qobs - Qsim).^2);
    denominator = sum((Qobs - obs_mean).^2) + sum((Qsim - sim_mean).^2);
    kg = 1 - numerator / denominator;
end

% function L = fkge(Q_obs, Q_sim)
%     %%%%%%%%%%%%%%%%%%%NSE%%%%%%%%%%%%%%%%%%%%%%%
%     % Q_obs_mean = mean(Q_obs);
%     % numerator = sum((Q_obs - Q_sim).^2);
%     % denominator = sum((Q_obs - Q_obs_mean).^2);
%     % L = 1 - (numerator / denominator);
%     %%%%%%%%%%%%%%%%%%%KGE%%%%%%%%%%%%%%%%%%%%%%%
%     Q_sim = Q_sim(:);
%     Q_obs = Q_obs(:);
% 
%     % 检查标准差
%     if std(Q_sim) == 0 || std(Q_obs) == 0
%         % warning('Standard deviation of Q_sim or Q_obs is zero. KGE calculation may result in NaN.');
%     end
% 
%     % 计算相关系数
%     r = corr(Q_sim, Q_obs);
% 
%     % 如果相关系数计算未定义，返回 NaN
%     if isnan(r)
%         % warning('Correlation calculation resulted in NaN. Check if Q_sim or Q_obs has sufficient variance.');
%         L = NaN;
%         return;
%     end
% 
%     % 计算标准差比
%     beta = std(Q_sim) / std(Q_obs);
% 
%     % 计算平均值比
%     gamma = mean(Q_sim) / mean(Q_obs);
% 
%     % 计算 KGE
%     L = 1 - sqrt((r - 1)^2 + (beta - 1)^2 + (gamma - 1)^2);
%     if isnan(L)
%         L=-99999;
%     end
% end