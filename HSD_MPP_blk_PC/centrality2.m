function [ inf_norm, mu ] = centrality2( blk, x, s, tau, kappa, nu )
% This function computes centrality measure by infinite norm.
% Centrality measure shows how close is a point to central path.

% 
hess_norm=0;
mu = (x'*s+tau*kappa)/nu;
for i = 1:length(blk)
    clear ph_i
    ind_i = sum(blk(1:i-1))+1:sum(blk(1:i));
    if mod(blk(i)-1,2) == 0
        gx(ind_i,1)= Hess_Ch_Even(x(ind_i), 'gradient');
    else
        gx(ind_i,1) = Hess_Ch_Odd(x(ind_i), 'gradient');
    end
end
    

inf_norm = norm([(s+mu*gx)/norm(gx,'inf'); tau*kappa-mu], 'inf');

end