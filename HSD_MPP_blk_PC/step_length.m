function [ alpha, alphaM, alphaP ] = step_length( blk, ...
     x, s, tau, kappa, dx, ds, dtau, dkappa, Phase )
% This function computes the step length for Block Moment Cone 
% and Non-Negative Polynomial Cone in Chebyshev
% Basis over [-1 1 ]
%   Detailed explanation goes here

k = length(blk);

if nargin <=9 
    Phase = 'conventional';
end

alphaM = zeros(1,k);
alphaP = zeros(1,k);
for i =1:k
    ind_i = sum(blk(1:i-1))+1:sum(blk(1:i));
    if mod(blk(i)-1,2) == 0
        alphaM(i) = alphaMom_Even(x(ind_i), dx(ind_i));
    else
        alphaM(i) = alphaMom_Odd(x(ind_i), dx(ind_i));
    end
    alphaP(i) = alphaPoly(s(ind_i), ds(ind_i), [-1 1]);
end


if dtau < 0
     alphatau = - tau/dtau;
else
     alphatau = 1e2;
end

if dkappa < 0
     alphakappa = - kappa/dkappa;
else
     alphakappa = 1e2;
end
    

if strcmp(Phase , 'predictor')
    alpha = .95*min([1, alphaM, alphaP, alphatau, alphakappa]);
elseif  strcmp(Phase , 'corrector')
    alpha = min([1, alphaM, alphaP, alphatau, alphakappa]);
elseif strcmp(Phase, 'conventional')
    alpha = .95*min([1, alphaM, alphaP, alphatau, alphakappa]);
end

% [min(alphaM), min(alphaP), alphatau, alphakappa, alpha]
    
end

