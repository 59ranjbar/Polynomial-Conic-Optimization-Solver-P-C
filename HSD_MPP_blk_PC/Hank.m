% Create Hankel Matrix of a Vector 
%Input: Vector c 
%Output: Hankel Matrix H
function H= Hank(c)

[m,n]= size(c);
if min(m,n)~= 1
    error('Input has to be a vector')
elseif n>1
    c = c';
end

if mod(length(c),2)~=1
    error('The length of vector c has to be odd')
end

l=length(c);
H=hankel(c(1:(l-1)/2+1),c((l-1)/2+1:l));

end
