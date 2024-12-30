function output=fun1(P,Pnew)
Pmin=min(P,Pnew);
Padd=max(Pnew-P,0);
Pminus=max(P-Pnew,0);
output=[Pmin,Padd,Pminus];