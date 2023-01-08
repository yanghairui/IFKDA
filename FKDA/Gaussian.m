function K= Gaussian( X,z )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
[gamma r] = size(X);
%gamma = size(X,1);
[m mm]=size(z);%%%z is the set of test samples
K=zeros(r,mm);
if gamma ~=m
    error('the dimension of input data is inconsistent!');
else
    for i = 1: r   
        for j=1:mm
            dis=X(:,i)-z(:,j); 
            K(i,j) = exp( -(   norm( dis)^2/ (gamma)) );   %一列代表的是原数据与一个测试样本的K
        end
    end
end
end

