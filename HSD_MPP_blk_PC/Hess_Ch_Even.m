function [ gxT, HxT ] = Hess_Ch_Even( xT, operand )
% This function computs gradient and Hessian of barrier function for
% Chebyshev Moment over interval [-1 1], this is, M^[-1 1]_Ch
% which is F(xT) = - ln(det((.5*(H+T))xT)) - ln(det(H+T)_Shiftted(xT))) where
% Shiftted(x) is
% Hank(xT(1:2*d-1) - .5*(xT(3:end) + [xT(3:-1:2); xT(1:2*d-3)])) +...
%      toeplitz(xT(1:d) - .5*(xT(3:d+2) + [xT(3:-1:2); xT(1:d-2)])) 
% 
%   Gradient is gxT = - deHT([(H+T)/2]_{-1}) where the inverse of H+T is
%   computed by drsolve package ((H+T)/2)_{-1} = thsolve(xT(1:d+1), xT(1:d+1)', xT(1:d+1), xT(d+1:end)', eye(d+1))

% Hessian will be computed by Type 1 Discrete Consine Transfer (DCT1) 
% Which is an implementation of paper 
% (Fast Polynomial Multiplication and Convolutions Related to the Discrete
% Consine Transform) by Cunter Baszenski 1997
%
% Input:  xT is the given point at which the gradient or/and Hessian will be
%               computed
% operand: tells to compute only gradient of both gradient and Hessian

% Output: gxT: gradeint of F(xT)
%             HxT: Hessain of F(xT)

% clear all
% n_i = 4;
% xT(1:2:n_i+1,1) =  2 ./ (1 - [0:2:n_i].^2)


if nargin < 2
    operand = 'hessian';
end


n = length(xT) - 1;
if mod(n,2) == 0
    d = fix(n/2);
else 
    error('Length of input vector is NOT even');
end

if size(xT,1) < size(xT,2)
    xT = xT';
end

if n <= 2
    xT_shift = [xT(1:n-1);0;0] - .5*([xT(3:end);0;0] + [xT(3:-1:2);xT(1:n-3);0;0]);
    error('Needs to be corrected');
else 
    xT_shift = xT(1:n-1) - .5*(xT(3:end) + [xT(3:-1:2);xT(1:n-3)]);
end


if d > 3
    HT1 = thsolve(xT(1:d+1),xT(1:d+1),xT(1:d+1),xT(d+1:end), eye(d+1));
    HT2 = thsolve(xT_shift(1:d), xT_shift(1:d), xT_shift(1:d), xT_shift(d:end), eye(d));
else
    HT1 = (hankel(xT(1:d+1), xT(d+1:end)) + toeplitz(xT(1:d+1))) \ eye(d+1);
    HT2 = (hankel(xT_shift(1:d), xT_shift(d:end)) + toeplitz(xT_shift(1:d))) \ eye(d);
end

deHT_HT1 = deHT(HT1);
deHT_HT2 = deHT(HT2);

gxT = -(deHT_HT1 + [deHT_HT2;0;0] - .5*([0;0;deHT_HT2] +...
    [deHT_HT2(3:end);0;0;0;0] + [0;deHT_HT2(2:-1:1);zeros(2*d-2,1)] ));
                          
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hessian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HxT =  [];                      
if strcmp(operand,'hessian') 
    B = sparse(eye(2*d-1, 2*d+1));
    B(2*(2*d-1)+1:2*d:end) = -.5;
    B(3:2*d:(2*d-1)*(2*d-3)) = -.5;
    B(2*d+1) = .5;
    B(4*d-1) = -1;

    HxT1 = zeros(n+1);
    HxT2 = zeros(n+1);
    for j = 0:n
        L1 = HT1*(HT(j,d))*HT1;
        HxT1(:,j+1) = deHT(L1);
        
        Adj2_DeHT = deHT(HT2*HankToep(B(:,j+1))*HT2);
        HxT2(:,j+1) = [Adj2_DeHT;0;0] - .5*([0;0;Adj2_DeHT] +...
            [Adj2_DeHT(3:end);0;0;0;0] + [0;Adj2_DeHT(2:-1:1);zeros(2*d-2,1)]);
    end 
    HxT = HxT1+HxT2;
    HxT = .5*(HxT+HxT');
    
%    [xT'*gxT  xT'*(HxT*xT)   gxT'*(HxT\gxT)]
end
   


end



%     HxT1 = zeros(n+1);
%     HxT2 = zeros(n-1);
%     for j = 0:n
%         L1 = HT1*(HT(j,d))*HT1;
%         HxT1(:,j+1) = deHT(L1);
%         if  j < n-1
%             L2 = HT2*(HT(j,d-1))*HT2;
%             HxT2(:,j+1) = deHT(L2);
%         end
%     end 
%     HxT = HxT1+B'*HxT2*B;


                          
























