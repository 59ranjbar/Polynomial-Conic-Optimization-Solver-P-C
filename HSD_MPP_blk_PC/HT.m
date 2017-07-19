function M = HT( k, m )
%This function creates m+1 by m+1 matrix
% where HTk(i,j)={1,  if   (i+j=k or |i-j|=k) 
%                        {0, if    o.w.

% clear all
% m = 2;
% k = 0;


 if 0 <= k <= m
     ee = [zeros(k,1); 1; zeros(2*m-k,1)];
     M = hankel(ee(1:m+1),ee(m+1:end)) + toeplitz(ee(1:m+1));
 elseif m < k <= 2*m
     ee = [zeros(k,1); 1; zeros(2*m-k,1)];
     M = hankel(ee(1:m+1),ee(m+1:end));
 end
 

end






