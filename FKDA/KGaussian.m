% function K = KGaussian(A, tilde_A, gamma)
function K = KGaussian(A)
% for building kernel data matrix, reduced or full, with Gaussian kernel. 
% 
% Inputs 
% A: full data set. 
% 
% Outputs 
% K: kernel data, full or reduced. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
[gamma,r] = size(A);K1=zeros(r,r);
for i = 1: r 
    for j = i: r
      dis=A(:,i)-A(:,j); 
      K1(i,j) = exp( -(   norm( dis)^2/ (gamma) )); 
    end 
end

K=K1'+K1-eye(r);
end