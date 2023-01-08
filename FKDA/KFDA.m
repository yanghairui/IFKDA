function [C,N,CLabel,c] = KFDA( X,XLabel )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[d,n]=size(X);
CLabel=unique(XLabel,'stable');
c=length(CLabel);  % the number of all sample classes  
N=zeros(1,c);
%%%%%%compute M
C=zeros(d,c);
%%%直接求C
% for i=1:c
k=1;
for i=1:c
    loc=[];
    loc=find(XLabel==CLabel(i));
    N(k)=length(loc);
    C(:,k)=mean(X(:,loc),2);
    k=k+1;
end
% M=zeros(n,c);
%%%求M，再求C
% for i=1:c
%     loc=[];
%     loc=find(XLabel==i);
%     N(i)=numel(loc);
%     m=zeros(n,1);
%     m(loc)=1/N(i);
%     M(:,i)=m;
% end
% C=X*M;

%%%computeK_C,c*c
%K_C=KGaussian(C); 
%K_C=Polynomial(C,C);
%K_C=sig(C,C,c);

%K_C_inv=inv(K_C);

end

