function [ alpha ] = alphaMom_Odd( x, dx )
% This function computes the step-length of given point and direction in
% M_{-1,1} with odd degree in Cheb Basis

n = length(x) - 1;
if mod(n,2) ~= 1
    error('Degree of x is NOT odd');
else
    d = fix(n/2);
end


x_shift1 = x(1:end-1)+.5*x(2:end)+.5*[x(2);x(1:end-2)];
dx_shift1 =  dx(1:end-1)+.5*dx(2:end)+.5*[dx(2);dx(1:end-2)];

chol_HT1 = chol(Hank(x_shift1) + toeplitz(x_shift1(1:d+1)));
M1 = (chol_HT1' \ (Hank(dx_shift1)+toeplitz(dx_shift1(1:d+1)))) / chol_HT1;
M1 = 0.5*(M1+M1');
Eig1 = eig(M1);    
alphaM1 = 1/max(-Eig1);
if alphaM1 <= 0
    alphaM1 = 1e2;
end


x_shift2 = x(1:end-1)-.5*x(2:end)-.5*[x(2);x(1:end-2)];
dx_shift2 =  dx(1:end-1)-.5*dx(2:end)-.5*[dx(2);dx(1:end-2)];

chol_HT2 = chol(Hank(x_shift2) + toeplitz(x_shift2(1:d+1)));
M2 = (chol_HT2' \ (Hank(dx_shift2)+toeplitz(dx_shift2(1:d+1)))) / chol_HT2;    
M2 = .5*(M2+M2');
Eig2 = eig(M2);
alphaM2 = 1/max(-Eig2);
if alphaM2 <= 0
    alphaM2 = 1e2;
end


alpha = min(alphaM1, alphaM2);



end

