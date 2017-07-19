function [ dx, dy, ds, dtau, dkappa ] = predir( blk,...
    x, s, tau, kappa, A, b, c, rp, rd, rg, mu)
        
% This function computes the Newton search direction in predictition phase
% With Regulrization technique. Saunders&Tomlin

gamma_sqr = 1e-8;
sigma_sqr = 1e-8;

k = length(blk);
m = length(b);
n = sum(blk);
schur = zeros(m);
rhsSchur = zeros(m,2);
gx = zeros(n,1);
Rd = rd+s;


for i = 1:k
    ind_i = sum(blk(1:i-1))+1:sum(blk(1:i));
    if mod(blk(i)-1,2) == 0
        [gx(ind_i,1), Hx{i}] = Hess_Ch_Even(x(ind_i), 'hessian');
    else
        [gx(ind_i,1), Hx{i}] = Hess_Ch_Odd(x(ind_i), 'hessian');
    end
    Hx{i} = mu*Hx{i} + gamma_sqr*eye(blk(i));  % Regularization
    Hx_inv{i} = Hx{i}\eye(blk(i));   
    Hx_inv_At{i} = Hx_inv{i}*A(:,ind_i)' ;
    schur = schur + A(:, ind_i)*Hx_inv_At{i};
    rhsSchur = rhsSchur + Hx_inv_At{i}'*[Rd(ind_i) , c(ind_i)];
end


schur = schur + sigma_sqr*eye(m);
rhsSchur = rhsSchur + [rp , b];

vq = schur\rhsSchur;
up = zeros(n,2);


clear ind_i
for i = 1:k
     ind_i = sum(blk(1:i-1))+1:sum(blk(1:i));
     up(ind_i,:) = Hx_inv_At{i}*vq - Hx_inv{i}*[Rd(ind_i) , c(ind_i)];
end



dtau = (c'*up(:,1) - b'*vq(:,1) + rg - kappa)/...
          (-c'*up(:,2) + b'*vq(:,2) + kappa/tau);

dy = vq(:,1)  + dtau*vq(:,2);
dx = up(:,1) + dtau* up(:,2);   
ds = rd - A'*dy + c*dtau;
dkappa = - kappa - (kappa/tau)*dtau;




end










