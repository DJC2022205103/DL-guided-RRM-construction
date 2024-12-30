function u = uh(n, K, T)
t=1:T;
coeff = 1 / (K * gamma(n));
u = coeff * (t / K).^(n-1) .* exp(-t / K);
end
