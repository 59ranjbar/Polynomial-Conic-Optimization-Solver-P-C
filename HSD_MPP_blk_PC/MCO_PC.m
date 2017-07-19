function [x_opt,y_opt,s_opt] = MCO_PC(A0,b0,c0,blk,scale_data,file_name)
% This script is an implementation of Non-Symmeteric 
% Homogeneous Self-Dual Predictor-Corrector
% Interior Point Method  
% for Multi-Block Moment Cone and its dual problem, this is,
% Univariate Non-Negative Polynomial 
% in Chebyshev Basis on interval  [-1 1].
% This is, min      c_1'*x_1+...+c_k'*x_k
%             s.t.      A_1*x_1+...+A_k*x_k = b,
%                               x_i  in  M^{n_i}_[-1 1]_Ch    i=1,...,k.
% where dual problem is 
%
%         max    b'*y
%          s.t.    A'*y + s_i = c_i,                       i=1,...,k
%                             s_i  in  P^{n_i^+}_[-1 1]_Ch
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Details
%   A = [A_1,...,A_k]
%   c = [c_1,...,c_k]
%   b the right hand side vector
%   scale_data = 0 if scaling is not needed and 1 otherwise
%   file_name = file's name to write the iterations information
%   blk = a vector of blocks' dimensions

%%%%%%%%%%%%%%%%%%%%%%%%% 
%Initialize Pars
%%%%%%%%%%%%%%%%%%%%%%%%%
gap_tol = 1e-6;              % duality gap tolerance
inf_tol = 1e-6;                % infeasibility tolerance
maxIter = 100;              % maximum number of iterations
iter = 0;                        % iteration counter for the main while loop
Stop = 0;                     % Stopping flag for main loop
beta =.9;                     % large Neig. parameter
eta = .45;                     % small Neig. parameter
iter_inner = 3;              % number of correction phase need in each iteration


% if nargin < 5
%     scale_data = 1 ;
% end
% 
% if nargin < 6
%     file_name = 'iter_MCO_MPC';
% end
%%%%%%%%%%%%%%%%%%%%%%
% Initial strictly feasible points for cones, nu, mu
%%%%%%%%%%%%%%%%%%%%%%
[ x0, y0, s0, tau0, kappa0, mu0, nu ] = initial( blk, b0 );

%%%%%%%%%%%%%%%%%%%%
% Scaling A, b, c
%%%%%%%%%%%%%%%%%%%%%
if scale_data == 1
    [A,b,c,normA,normc,normb] = scaling(blk,A0,b0,c0);  
else
     A = A0; b = b0; c = c0; normb = 1; normc = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%
% Initial Residuals
%%%%%%%%%%%%%%%%%%%%%%%
x = x0; y = y0; s = s0; kappa = kappa0; tau = tau0; mu = mu0;
rp = tau*b - A*x;
rd = tau*c - A'*y - s;  
rg = kappa + c'*x - b'*y;

%%%%%%%%%%%%%%%%%%%%%%%
%Print date and the Initial information 
%%%%%%%%%%%%%%%%%%%%%%% 
file_id = fopen(file_name , 'a');                    % where to write the iterations
fprintf(file_id, '|Ax-b|= %.2e, |A''y+s-c|= %.2e, |b''y-c''x|= %.2e, x''s= %.2e \n',...
    norm(A0*x0-b0), norm(A0'*y0+s0-c0), abs(b0'*y0-c0'*x0), x0'*s0);
fprintf(file_id, ' k: r   rp            rd            |cx-by|     |x''*s|         mu            alpha_a     alpha_c    tau         kappa \n');

%%%%%%%%%%%%%%%%%%%%%%%%%
% Main loop
%%%%%%%%%%%%%%%%%%%%%%%%%
while(~Stop && iter < 19) % maxIter) 
	iter = iter +1;
    %%%%%%%%%%%%%%%%%%%%%%%%
    % Prediction Phase 
    %%%%%%%%%%%%%%%%%%%%%%%%
    [ dx_p, dy_p, ds_p, dtau_p, dkappa_p ] = predir( blk,...
        x, s, tau, kappa, A, b, c, rp, rd, rg, mu);

    %%%%%%%%%%%%%%%%%%%%%%%%%
    % Step-Length in Affine Phase 
    %%%%%%%%%%%%%%%%%%%%%%%%%
    [alpha_p_pos(iter), alphaM_p, alphaP_p] = step_length( blk,...
        x, s, tau, kappa, dx_p, ds_p, dtau_p, dkappa_p, 'predictor' );

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Find step-length that z_alpha is in Neib
    %%%%%%%%%%%%%%%%%%%%%%%%%%     
    al=[]; j=0; inf_norm_al_p=[]; mu_al_p=[];
    for al =alpha_p_pos(iter):-.1:0
        j = j+1;
        [inf_norm_al_p(j), mu_al_p(j)] = centrality2(blk,...
            x+al*dx_p, s+al*ds_p, tau+al*dtau_p, kappa+al*dkappa_p, nu);
    end
    
    if isempty(find(inf_norm_al_p <= beta*mu_al_p,1))
        alpha_p(iter) = 0;
    else
        alpha_p(iter) = alpha_p_pos(iter) - .1*(find(inf_norm_al_p <= beta*mu_al_p,1)-1);
    end
    
    
    x_p         = x + alpha_p(iter)*dx_p;
    y_p         = y + alpha_p(iter)*dy_p;
    s_p         = s + alpha_p(iter)*ds_p;
    tau_p      = tau + alpha_p(iter)*dtau_p;
    kappa_p = kappa + alpha_p(iter)*dkappa_p;                                        
    
    mu_p = (x_p'*s_p + tau_p*kappa_p)/nu;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Corrector Phase
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
   for r = 1: iter_inner
        [dx_c, dy_c, ds_c, dtau_c, dkappa_c] = corrdir( blk,...
            x_p, s_p, tau_p, kappa_p, A, b, c, mu_p);

        %%%%%%%%%%%%%%%%%%%%%%%%%
        % Step Length in Correction Phase 
        %%%%%%%%%%%%%%%%%%%%%%%%%
        [alpha_c_pos(r,iter), alphaM_c, alphaP_c] = step_length( blk,...
            x_p, s_p, tau_p, kappa_p, dx_c, ds_c, dtau_c, dkappa_c, 'corrector' );

        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Find step-length that z_alpha is in Neib
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        al=[]; j=0; inf_norm_al_c=[]; hess_norm_al_c=[]; mu_al_c=[];
        for al = alpha_c_pos(r,iter):-.1:0
             j = j+1;
             [inf_norm_al_c(j), mu_al_c(j)] = centrality2(blk, ...
                 x_p+al*dx_c, s_p+al*ds_c, tau_p+al*dtau_c, kappa_p+al*dkappa_c, nu); 
%              [inf_norm_al_c(j), hess_norm_al_c(j), mu_al_c(j)] = centrality(blk, ...
%                  x_p+al*dx_c, s_p+al*ds_c, tau_p+al*dtau_c, kappa_p+al*dkappa_c, nu); 
        end


%         [min_hess_norm_al_c, ind] = min(hess_norm_al_c);
        [min_inf_norm_al_c, ind] = min(inf_norm_al_c); 
        alpha_c(r,iter) = alpha_c_pos(r, iter) - (ind-1)*.1;
        
         if  inf_norm_al_c(ind) <= eta*mu_al_c(ind) ||  r == iter_inner 
            break;
        else
            x_p         = x_p + alpha_c(r,iter)*dx_c;
            y_p         = y_p + alpha_c(r,iter)*dy_c;
            s_p         = s_p + alpha_c(r,iter)*ds_c;
            tau_p      = tau_p + alpha_c(r,iter)*dtau_c;
            kappa_p = kappa_p + alpha_c(r,iter)*dkappa_c;                                        

            mu_p = (x_p'*s_p + tau_p*kappa_p)/nu; 
        end
   end

    
    %%%%%%%%%%%%%%%%%%%%%%
    % Prepare points for next iteration
    %%%%%%%%%%%%%%%%%%%%%%
    x         = x_p + alpha_c(r,iter)*dx_c;
    y         = y_p + alpha_c(r,iter)*dy_c;
    s         = s_p + alpha_c(r,iter)*ds_c;
    tau      = tau_p + alpha_c(r,iter)*dtau_c;
    kappa = kappa_p + alpha_c(r,iter)*dkappa_c;                                        

    rp = tau*b - A*x;
    rd =  tau*c - A'*y - s;
    rg = kappa + c'*x - b'*y;

    mu = (x'*s + tau*kappa) / nu;
        
    %%%%%%%%%%%%%%%%%%%%%%%%
    % Chech stopping criteria 
    %%%%%%%%%%%%%%%%%%%%%%%%
    gap(iter) = (normb*normc/tau^2)*(x'*s); 
    obj(:,iter) = (normb*normc/tau)*[c'*x ;  b'*y]; 
    obj_gap = obj(1,iter) - obj(2,iter);
    
    rel_gap = gap(iter)/max(1,mean(abs(obj(:,iter))));

    pri_inf  = norm(rp)/tau; 
    dual_inf = norm(rd)/tau; 
    inf_meas(iter) = max(pri_inf, dual_inf); 
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    % Print Iteration's Information
    %%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf(file_id, '%2d:%1d  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e \n',...
                        iter, r,  pri_inf, dual_inf, rel_gap, abs(gap(iter)),...
                        mu/tau^2, alpha_p(iter), alpha_c(r,iter), tau, kappa ); 

    %%%%%%%%%%%%%%%%%%%%%%%%
    % Check the Stopping Criteria
    %%%%%%%%%%%%%%%%%%%%%%%%      
    if  max(rel_gap, inf_meas(iter)) <= 1e2*gap_tol
        Stop = 1;
         if scale_data ==1 
             for i = 1:length(blk)
                 ind_i = sum(blk(1:i-1))+1:sum(blk(1:i));

                 x_opt(ind_i,1) = x(ind_i)*(normb/(normA(i)*tau));
                 s_opt(ind_i,1) = s(ind_i)*(normc*normA(i)/tau);  
             end
             y_opt = y*(normc/(tau)); 
         else 
             x_opt = x/tau;
             y_opt = y/tau; 
             s_opt = s/tau;
         end
         fprintf(file_id, 'optimal points has been reached \n');
         fprintf(file_id, '|Ax-b|= %.2e, |A''y+s -c|= %.2e, |b''y-c''x|= %.2e, x''s= %.2e, mu= %.2e \n',...
            norm(A0*x_opt-b0), norm(A0'*y_opt+s_opt-c0), abs(b0'*y_opt-c0'*x_opt), x_opt'*s_opt, mu/tau^2);
    elseif max(mu / mu0 , (tau / kappa) / (tau0 /kappa0 )) <1e3*eps
        Stop = -1 ;
        fprintf(file_id, 'Infeasibility has been detected \n')
    end    

    
end

fprintf(file_id, '------------------------------------------------- \n')
fclose(file_id);


end






