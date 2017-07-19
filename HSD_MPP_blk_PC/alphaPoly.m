function [ alpha, flag ] = alphaPoly( s, ds, I )
% This function computes step length in Non-negative polynomial cone in Chebyshev Basis, This
% is given s in int(P+_Ch) and direction ds this function finds an alpha such
% that s+alpha*ds in (P+_Ch) over interval I

% Input:   s in int(P+_Ch)
%           ds direction
%              I: Interval
%Output:  alpha: step length 
%                flag: show situation 

%  This is a discretized method, this is, we consider p points, (t_i), in the interval I 
%  and take alpha such that in all those points s(t_i)+alpha*ds(t_i) >= 0

if nargin < 3 
    a = 0;
    b = 1;
elseif  nargin == 3
    a = I(1,1);
    b = I(1,2);
end

p = 500*length(s);  % # of points in interval
t  = linspace(a, b, p);

ds_val = chebpolval(ds,t);
s_val =  chebpolval(s,t);

minmax_ds = minmax(ds_val);
minmax_s = minmax(s_val);


if   minmax_ds(1,1) > 0
    alpha = 1e2;
    flag = -1;
elseif  minmax_ds(1,2) < -1e-8
    alpha = min(s_val ./ abs(ds_val));
    flag = 1;
elseif   minmax_s(1,2) < 0
    alpha = 0;
    flag = 0;
else
    alpha = min(s_val(ds_val < -1e-12) ./ abs(ds_val(ds_val < -1e-12)));
    flag = 1;
end

end

