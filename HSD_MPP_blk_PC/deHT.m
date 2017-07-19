function [ x ] = deHT( Z )
% This function computes the dehank+detoeplitz of a matrix, this is <(Hi+Ti), Z>
%   Input: matrix Z
%   Output: vector x

[m, d] = size(Z);
if m~=d
    error('Input matrix is not square');
end

% if Z~= Z'
%     error('Matrix is not symmetric');
% end

Z_flipped = fliplr(Z); 
x = zeros(2*d-1,1);

x(1,1) = sum(diag(Z_flipped, d-1)) + sum(diag(Z, 0));
for i = 1:d-1
    x(i+1,1) = sum(diag(Z_flipped, d-1-i)) + sum(diag(Z, i)) + sum(diag(Z, -i));
end

for i = d:2*d-2
    x(i+1,1) = sum(diag(Z_flipped, d-1-i)) ;
end

end