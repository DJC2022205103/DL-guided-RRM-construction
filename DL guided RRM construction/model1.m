function qsim=model1(P,qobs,para,W)
D=para(1);
EP=para(2);
WM=para(3);
B=para(4);
FC=para(5);
CG=para(6);
N=para(7);
K=para(8);
L=para(9);
% a=para(7);
% b=para(8);
% h=para(9);
% d=para(10);
% w=para(11);

T=length(P);
for t=1:T
    if t==1
        Pnew(1)=(1-L)*P(1);
    else
        Pnew(t)=(1-L)*P(t)+L*P(t-1);
    end
end
P=Pnew;
UH=uh(N,K,T);
% UH=uh2(a,b,h,d,w,T);
WMM = (1 + B) * WM;
sp0=0;
PD=zeros(1,T);
RS=zeros(1,T);
RG=zeros(1,T);
QG=zeros(1,T);
QG0=qobs(1);
RR=zeros(1,T);
WW=zeros(1,T);
dt=1;
F=57.3;
W=min(W,WM);
for t=1:T
    %净雨
    sp=sum(P(1:t));
    if sp0>=D
        PD(t)=P(t);
    elseif sp<=D
        PD(t)=0;
    else
        PD(t)=sp-D;
    end
    sp0=sp;
    PD(t)=max(0,PD(t)-EP);

    %产流
    if PD(t) > 0
        if WM - W <= 0
            A = WMM;
        else
            A = WMM * (1 - (1 - W / WM)^(1 / (B + 1)));
        end
        if PD(t) + A < WMM
            R = (PD(t) - WM + W + WM * (1 - (PD(t) + A) / WMM)^(1 + B));
        else
            R = (PD(t) - WM + W);
        end
    else
        R = 0;
    end
    RR(t)=R;
    W=W+PD(t)-R;
    WW(t)=W;

    %分水源
    if PD(t)==0
        RG(t)=0;
        RS(t)=0;
    else
        RG(t)=R*min(PD(t),FC)/PD(t);
        RS(t)=R-RG(t);
    end

    %汇流
    QG(t) = QG0 * CG + RG(t) * (1 - CG) * F / 3.6 / dt;
    QG0 = QG(t);
end
temp=conv(RS,UH)*F/3.6/dt;
QS=temp(1:T);
QS=QS(:);
QG=QG(:);
qsim=QS+QG;