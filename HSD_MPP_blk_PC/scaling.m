function [A,b,c,normA,normc,normb,x0,y0,s0] = scaling(blk,A,b,c) % ,x0,y0,s0
% This function scales input data
%
%
%
%
%


m = length(b); 
numblk = length(blk);   %number of blocks;  

normA = zeros(1,numblk);
normc = 0;
for i = 1:numblk
    ind_i = sum(blk(1:i-1))+1:sum(blk(1:i));
    
    normA(i) = max(1, sqrt(sum(sum(A(:,ind_i).*A(:,ind_i)))));
    normc = max(normc,norm(c(ind_i),'fro'));           %% fro norm
end
normc = max(1,normc);


clear ind_i

for i = 1:numblk
    ind_i = sum(blk(1:i-1))+1:sum(blk(1:i));

    A(:,ind_i) = A(:,ind_i)/normA(i);
    c(ind_i)  = c(ind_i)/(normc*normA(i)); 
    
    if (nargin == 7)
        x0(ind_i) = x0(ind_i)*normA(i); 
        s0(ind_i) = s0(ind_i)/(normc*normA(i)); 
	end
end

normb = max(1,norm(b));
b = b/normb;


end

