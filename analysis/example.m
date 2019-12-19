%%%Optimal Control in Networks
%%%Last by Khizar, 2019-12-18

%% Here we will solve a simple case in which we let x, b, w, g, p denote the opinion, trust, constant weight, loss, costate respectively.

%Symbolic instantiation
syms p w x g b

%Environment
N = 100;
w = 1 / N;
g = x^2;
beta = 3;

%Learning function
M(x,b) =  1/(1+((x/(1-x))^(-b)));
dMdx = diff(M,x);

%Sanity checks
disp(M(0.5,2));
disp(dMdx(.5, 2));
thetas = linspace(0.01, 0.99, 100);

%Function plot
m1 = M(thetas,beta);
plot(thetas, m)

%Function derivative plot
md = dMdx(thetas, beta);
plot(thetas, md)

%Hamiltonian
H = p * w * (dMdx(x, beta) * dMdx(x-u, beta)) + int(g);

%Gradient

dHdp = -diff(H,p);
dHdx = diff(H,x);
dHdu = diff(H,u);

%Constraints
assume(x>0);
assume(x<1);

%Solve
y=solve(dHdu,u);
disp(y)
