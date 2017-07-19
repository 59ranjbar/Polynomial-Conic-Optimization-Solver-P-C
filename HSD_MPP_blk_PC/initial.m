function [ x, y, s, tau, kappa, mu, nu ] = initial( blk, b0 )
% This function constructs initial points that are on the centeral paths, 
% this is, || s+mu*gx ||^*_x = 0

m = length(b0);
k = length(blk);
nu = sum(blk)+1;

for i = 1:k
    if blk(i) ~= 0
        n_i = blk(i)-1;
        ind_i = sum(blk(1:i-1))+1:sum(blk(1:i));
        x(ind_i,1) = zeros(n_i+1,1);
        x(ind_i(1:2:n_i+1)) =  2 ./ (1 - [0:2:n_i].^2);
        if mod(n_i,2) == 0
            s(ind_i,1) = - Hess_Ch_Even(x(ind_i),'gradient');
        else
            s(ind_i,1) = - Hess_Ch_Odd(x(ind_i),'gradient');
        end
    end
end

y = zeros(m,1);
tau = 1;
kappa = 1;

mu = (x'*s+tau*kappa)/nu;

end

