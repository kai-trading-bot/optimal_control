%% One bot 1 person
syms p w x g b u
M(x,b) =  1/(1+((x/(1-x))^(-b))); %magic function
dMdx = diff(M,x);
g = x^2;
b=3;

%Hamiltonian
H = 2* p * w * (dMdx(x, b) * dMdx(x-u, b)) + int(g);
assume(x>0);
assume(x<1);

%costate
dHdu = - diff(H,u); %dp1

y=solve(dHdu,u);


%% Two bot 1 person
syms p1 p2 w x g b u1 u2 beta

M(x,b) =  1/(1+((x/(1-x))^(-b))); %magic function
dMdx = diff(M,x);
g = x^2;
b=3;

%Hamiltonian
H = w * (dMdx(x, b) * p1* dMdx(x-u1, b) * p2 * dMdx(x-u2, b)) + int(g);

%Costate equations
dHdu1 =  diff(H,u1); %dp1
dHdu2 =  diff(H,u2); %dp2

assume(x>0);
assume(x<1);
assume(u1<1);
assume(u1>0);
assume(u2<1);
assume(u2>0);
%control
dHdx = diff(H,x);
%solve(dHdx,x)

y1=solve(dHdu1,u1);
y2=solve(dHdu2,u2);


%% Three bot 1 person
syms p1 p2 p3 w x g b u1 u2 u3 beta

M(x,b) =  1/(1+((x/(1-x))^(-b))); %magic function
dMdx = diff(M,x);
g = x^2;
b=3;

%Hamiltonian
H = w * (dMdx(x, b) * (p1* dMdx(x-u1, b)) + (p2 * dMdx(x-u2, b)) +  (p3 * dMdx(x-u3, b))) + int(g);
% add covariance?
%Costate equations
dHdu1 =  diff(H,u1); %dp1
dHdu2 =  diff(H,u2); %dp2
dHdu3 =  diff(H,u3); %dp2

assume(x>0);
assume(x<1);
assume(u1<1);
assume(u1>0);
assume(u2<1);
assume(u2>0);
assume(u3<1);
assume(u3>0);
%control
dHdx = diff(H,x);
%solve(dHdx,x)

y1=solve(dHdu1,u1);
y2=solve(dHdu2,u2);
y3=solve(dHdu3,u3);





%% Two person One bot
syms p1 p2 w x x1 x2 g1 g2 b u beta

M(x,b) =  1/(1+((x/(1-x))^(-b))); %magic function
dMdx = diff(M,x);
g1 = x1^2;
g2 = x2^2;
b=2;

%Hamiltonian ... this needs to be for looped they affect each other but
%become unsolvable
H1 = w * (dMdx(x1, b) * p1* dMdx(x1-u, b) * p2 * dMdx(x1-x2, b)) + int(g1);
H2 = w * (dMdx(x2, b) * p1* dMdx(x2-u, b) * p2 * dMdx(x2-x1, b)) + int(g2);

dH1dx = diff(H1,x1);
dH1dp = diff(H1,p1);
dH2dx = diff(H2,x2);
dH2dp = diff(H2,p2);

assume(x>0);
assume(x<1);
% dsolve(dH1dx,dH1dp,dH2dx,dH2dp)

dH1du = diff(H1,u);
solve(dH1du,u);