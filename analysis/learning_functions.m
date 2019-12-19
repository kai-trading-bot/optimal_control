%%%Learning functions
%%%Last by Khizar, 2019-12-18
%% In this file, we provide various activation functions for learning

% Default parameters
alpha = 0.5;
beta = 2;

% Sigmoid learning
S(x) = 1 / (1 + exp(-x));

% Tanh
T(x) = (2 / (1 + exp(-2 * x))) - 1;

% Binary
B(x) = max(0, 1);

% Arctan
A(x) = atan(x);

% ReLU
R(x) = max(0, x);

% PReLU
P(x) = ((x < 0) * alpha * x) + ((x >= 0) * x);

% ELU
E(x) = ((x < 0) * alpha * (exp(x) - 1)) + ((x >= 0) * x);

% SoftPlus
P(x) = log(1 + exp(x));

% Trust learning
M(x,b) =  1/(1+((x/(1-x))^(-b)));

