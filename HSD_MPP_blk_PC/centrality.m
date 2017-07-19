function [ inf_norm, hess_norm, mu ] = centrality( blk, x, s, tau, kappa, nu )
% This function computes centrality measure by infinite norm.
% Centrality measure shows how close is a point to central path.

% 
hess_norm = 0;
mu = (x'*s+tau*kappa)/nu;


for i = 1:length(blk)
    clear ph_i
    ind_i = sum(blk(1:i-1))+1:sum(blk(1:i));
    if mod(blk(i)-1,2) == 0
        [gx(ind_i,1), Hx{i}] = HessT_DCT1_Int(x(ind_i), 'hessian');
    else
        [gx(ind_i,1), Hx{i}] = Hess_Ch_Odd(x(ind_i), 'hessian');
    end
    
    ph_i = s(ind_i)+mu*gx(ind_i);
    hess_norm = hess_norm+ph_i'*(Hx{i}\ph_i);
end
    

inf_norm = norm([(s+mu*gx)/norm(gx,'inf'); tau*kappa-mu], 'inf');
hess_norm = (hess_norm+(tau*kappa - mu)^2)^.5;

end

