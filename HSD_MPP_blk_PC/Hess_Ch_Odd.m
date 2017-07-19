function [ gxT, HxT ] = Hess_Ch_Odd( xT, operand )
% This function computs gradient and Hessian of barrier function for
% Chebyshev Moment of Odd degree over interval [-1 1],  M_{[-1,1]_Ch}
% which is F(xT) = - ln det(H+T)_xT_shift1  -  ln det(H+T)_xT_shift2
%
% x_shift1 is x(1:2*n+1) + .5*x(2:2n+1) + .5*[x(2); x(1:2n)] in H
%                      x(1:n+1) + .5*(x(2:n+2) + .5*[x(2); x(1:n)]))    in T
%                      
% and
% x_shift2 is x(1:2*n+1) - .5*x(2:2n+1) - .5*[x(2); x(1:2n)] in H
%                      x(1:n+1) - .5*x(2:n+2) - .5*[x(2); x(1:n)]        in T
%
%
%   Gradient is gxT = - deHT([(H+T)/2]_{-1}) where the inverse of H+T is
%   computed by drsolve package ((H+T)/2)_{-1} = thsolve(xT(1:d+1), xT(1:d+1)', xT(1:d+1), xT(d+1:end)', eye(d+1))

% Hessian will be computed by Type 1 Discrete Consine Transfer (DCT1) 
% Which is an implementation of paper 
% (Fast Polynomial Multiplication and Convolutions Related to the Discrete
% Consine Transform) by Cunter Baszenski 1997
%
% Input:     xT: in M_{[-1,1]}_{CH} of Odd degree (R^{2n+2})
% Output: gxT: gradeint of F(x)
%             HxT: Hessain of F(x)

if nargin < 2
    operand = 'hessian';
end



n = length(xT) - 1;
if  mode(n,2) ~= 1
    d = fix(n/2);
else
    error('xT is NOT of odd degree')
end

if size(xT,1) < size(xT,2)
    xT = xT';
end

xT_shift1 = xT(1:end-1)+.5*xT(2:end)+.5*[xT(2);xT(1:end-2)];
xT_shift2 = xT(1:end-1)-.5*xT(2:end)-.5*[xT(2);xT(1:end-2)];

if d >= 3
    HT1 = thsolve(xT_shift1(1:d+1), xT_shift1(1:d+1), xT_shift1(1:d+1), xT_shift1(d+1:end), eye(d+1));
    HT2 = thsolve(xT_shift2(1:d+1), xT_shift2(1:d+1), xT_shift2(1:d+1), xT_shift2(d+1:end), eye(d+1));
else
    HT1 = (hankel(xT_shift1(1:d+1), xT_shift1(d+1:end)) + toeplitz(xT_shift1(1:d+1))) \ eye(d+1);
    HT2 = (hankel(xT_shift2(1:d+1), xT_shift2(d+1:end)) + toeplitz(xT_shift2(1:d+1))) \ eye(d+1);
end

HT1_deHT = deHT(HT1);
HT2_deHT = deHT(HT2);
gxT = - ([HT1_deHT;0] + .5*[0;HT1_deHT] +...
            .5*[HT1_deHT(2);HT1_deHT(1)+HT1_deHT(3);HT1_deHT(4:end);0;0] +...
            [HT2_deHT;0] - .5*[0;HT2_deHT] -...
            .5*[HT2_deHT(2);HT2_deHT(1)+HT2_deHT(3);HT2_deHT(4:end);0;0]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hessian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HxT =  [];
if strcmp(operand,'hessian') 
    A1 = sparse(eye(n,n+1));
    A1(n+1:n+1:end) = .5;
    A1(2:n+1:end-n) = .5;
    A1(1,2) = 1;

    A2 = sparse(eye(n,n+1));
    A2(n+1:n+1:end) = -.5;
    A2(2:n+1:end-n) = -.5;
    A2(1,2) = -1;
    
    for i = 0:n
        Adj1_deHT = deHT(HT1*HankToep(A1(:,i+1))*HT1);
        HxT1(:,i+1) = [Adj1_deHT;0] + .5*[0;Adj1_deHT] +...
            .5*[Adj1_deHT(2);Adj1_deHT(1)+Adj1_deHT(3);Adj1_deHT(4:end);0;0];
    
        clear Adj1_deHT
    end
    HxT1 = .5*(HxT1+HxT1');
    
    for i = 0:n
        Adj2_deHT = deHT(HT2*HankToep(A2(:,i+1))*HT2);
        HxT2(:,i+1) = [Adj2_deHT;0] - .5*[0;Adj2_deHT] -...
            .5*[Adj2_deHT(2);Adj2_deHT(1)+Adj2_deHT(3);Adj2_deHT(4:end);0;0];
    
        clear Adj2_deHT    
    end
    HxT2 = .5*(HxT2+HxT2');
    
    %%%%%%%%%%%%%%%%%%%%%%%
    % Hessian
    %%%%%%%%%%%%%%%%%%%
    HxT = HxT1+HxT2;
end    


    
end





% A1 = eye(n,n+1);
% A1(n+1:n+1:end) = .5;
% A1(2:n+1:end-n) = .5;
% A1(1,2) = 1;
% 
% A2 = eye(n,n+1);
% A2(n+1:n+1:end) = -.5;
% A2(2:n+1:end-n) = -.5;
% A2(1,2) = -1;
% 
% gxT_A = -(A1'*deHT(HT1) + A2'*deHT(HT2))  
% 
% for i = 0:n
%     HxT1(i+1,:) = A1'*deHT(HT1*HankToep(A1(:,i+1))*HT1);
% end
% for i = 0:n
%     HxT2(i+1,:) = A2'*deHT(HT2*HankToep(A2(:,i+1))*HT2);
% end
% HxT = HxT1+HxT2;

% H11 = Hank(x(1:end-1)+.5*x(2:end)+.5*[x(2);x(1:end-2)]) +...
%     toeplitz(x0(1:d+1)+.5*x0(2:d+2)+.5*[x0(2);x0(1:d)]) ;
% 
% H22 = Hank(x(1:end-1)-.5*x(2:end)-.5*[x(2);x0(1:end-2)]) +...
%     toeplitz(x(1:d+1)-.5*x(2:d+2)-.5*[x(2);x(1:d)])  ;



















