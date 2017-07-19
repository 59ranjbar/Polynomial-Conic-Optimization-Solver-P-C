function [ alpha ] = alphaMom_Even(x, dx)
% This function computes the step-length of given point and direction in
% M_{-1,1} with Even degree in Cheb Basis

n = length(x) - 1;
if mod(n,2) ~= 0
    error('Degree of x is NOT even');
else
    d = fix(n/2);
end



chol_H1 = chol(Hank(x)+toeplitz(x(1:d+1)));
M1 = (chol_H1' \ (Hank(dx) + toeplitz(dx(1:d+1)))) / chol_H1;
M1 = 0.5*(M1+M1');
Eig1 = eig(M1);
alphaM1 = 1/max(-Eig1);
if alphaM1 <= 0
    alphaM1 = 1e2;
end


chol_H2 = chol(Hank(x(1:2*d-1) - .5*(x(3:end) + [x(3:-1:2); x(1:2*d-3)])) +...
                         toeplitz(x(1:d) - .5*(x(3:d+2) + [x(3:-1:2); x(1:d-2)])));                    
M2 = (chol_H2' \ (Hank(dx(1:2*d-1) - .5*(dx(3:end) + [dx(3:-1:2); dx(1:2*d-3)])) +...
                            toeplitz(dx(1:d) - .5*(dx(3:d+2) + [dx(3:-1:2); dx(1:d-2)])))) / chol_H2;
M2 = .5*(M2+M2');
Eig2 = eig(M2);
alphaM2 = 1/max(-Eig2);
if alphaM2 <= 0
    alphaM2 = 1e2;
end


alpha = min(alphaM1, alphaM2);


end

