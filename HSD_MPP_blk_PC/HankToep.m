function HT = HankToep( x )
% This function constructs Hankel + Toeplitz matrix of a given vector
% 
% Input:  x given vector
% Output: HT the Hankel+Toeplitz of vector x

n = length(x);

if mod(n,2) == 0
    error('Length of vector should be 2*d+1');
end

d = (n-1) / 2;

HT = hankel(x(1:d+1), x(d+1:end)) + toeplitz(x(1:d+1));

end

