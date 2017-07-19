% This script constructs a feasible Maximum Flows Problem
% With capacities as non-negative polynomials,
% From the Network given in excel sheet and 
% then constructs the Dual problem which is a MCO
% Then this problem can be solved by MCO or CVX
%
%
% %%%%%%%%%%%%%%%%%%%%%%%%
% % Maximum Flows Problem Formulated as MCO
% %%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
%%%%%%%%%%%%%%%%%%%%%%
% Adjacent Matrix Of The Network Flows
%%%%%%%%%%%%%%%%%%%%%
Adj = xlsread('MaxFlowTestProb.xlsx','E2N1') ;   % The Network is
      % is given in excel sheet 'MaxFlowTestProb.xlsx'
      % Change the excel sheet name for different Network

edge = 2;             % number of edges
edge_s = 1;          % number of edges coming out from source node
node = 1;              % number of nodes without source and sink nodes
deg = 11;              % degree of poly with considering the constant 
                            % (actual degree is deg-1)

Adj = Adj(2:node+1,2:edge+1);
Adj(isnan(Adj)) = 0;

%%%%%%%%%%%%%%%%%%%%
% Capacities in Chebyshev basis in [-1,1]
%%%%%%%%%%%%%%%%%%%%
for i = 1:edge
    dd = randi([3,5]);
    P(:,i) = [4; zeros(dd,1); 1; zeros(deg-dd-2,1)];  % generating Non-Negative 
                                                    % Polynomial in Chebyshev basis in [-1,1]
end
P = [P; zeros(deg-size(P,1),size(P,2))];


%%%%%%%%%%%%%%%%%%%%%%
% Constructing b0, c0, A0, blk for MCO 
%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(deg,1);   
u(1:2:deg) = 2 ./ (1 - (0:2:deg-1).^2) ; 

bb = [ones(edge_s,1); zeros(edge-edge_s,1)];
b0 = kron(bb, u);

cc = zeros(2*node,1);
c0 = [kron(cc, zeros(deg,1)); reshape(P,[],1); kron(zeros(edge,1), zeros(deg,1))];

AA = zeros(size(Adj,2), 2*node);
AA(:,1:2:end) = Adj';
AA(:,2:2:end) = -Adj';
AA = [AA  eye(edge)  -eye(edge)];
A0 = kron(AA, eye(deg));
[m,n] = size(A0);

k = size(AA,2);
blk = deg*ones(1,k);
%%%%%%%%%%%%%%%%%%%%%%
% Solve By MCO_MPC
%%%%%%%%%%%%%%%%%%%%%
[x_opt,y_opt,s_opt] = MCO_PC(A0,b0,c0,blk,1,'iter');


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Solving MCO by means of CVX 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%
cvx_begin sdp 
    cvx_solver sedumi

    variable  x_cvx(length(c0),1)
    dual  variable  y_cvx

    minimize(c0'*x_cvx)
    y_cvx: A0*x_cvx == b0; 

    for i = 1:k
        n_i = blk(i)-1;
        d = fix(n_i/2);
        ind = sum(blk(1:i-1))+1:sum(blk(1:i));    
        x_i = x_cvx(ind);
        if  mod(n_i,2) == 0
            Hank(x_i) + toeplitz(x_i(1:d+1,1)) >= 0;
            Hank(x_i(1:2*d-1,1) - .5*(x_i(3:end,1) +...
                    [x_i(3:-1:2,1); x_i(1:2*d-3,1)])) +...
            toeplitz(x_i(1:d,1) - .5*(x_i(3:d+2,1) +...
                    [x_i(3:-1:2,1); x_i(1:d-2,1)])) >= 0;
        else
            x_cvx_shift1 = x_i(1:end-1)+.5*x_i(2:end) +...
                                          .5*[x_i(2); x_i(1:end-2)];
            x_cvx_shift2 = x_i(1:end-1)-.5*x_i(2:end) -...
                                          .5*[x_i(2); x_i(1:end-2)];

            hankel(x_cvx_shift1(1:d+1),x_cvx_shift1(d+1:end)) +...
                toeplitz(x_cvx_shift1(1:d+1)) >= 0;
            hankel(x_cvx_shift2(1:d+1),x_cvx_shift2(d+1:end)) +...
                toeplitz(x_cvx_shift2(1:d+1)) >= 0;
        end
    end
    
cvx_end
s_cvx = c0 - A0'*y_cvx; 
























