function [new_K_C,new_K_C_inv,N,new_C,new_Ytr,c] = Inc_KFDA(K_C,N,a,C,Ytr)
%ÿ�ζ���chunk%%%
%   Detailed explanation goes here_
old_N=N;
old_c=length(old_N);
%new_X=[X,a.train.X];
%new_XLabel=[XLabel,a.train.y];
%%%����N��C%%%%
%%%%%%%%����FLDA����������
new_C=C;
y=unique(a.train.y,'stable');
new_Ytr=Ytr;
for i=1:size(y,2)
    %%%�ȸ�����֪���%%%%
    %N(unique(a{1}.y))=N(unique(a{1}.y))+length(a{1}.y)/length(unique(a{1}.y));
    loc1=find(a.train.y==y(i));
    if ismember(y(i),Ytr)
        loc=find(Ytr==y(i));
        N(loc)=N(loc)+length(loc1);
        new_C(:,loc)=(old_N(loc)*new_C(:,loc)+sum(a.train.X(:,loc1),2))/N(loc);
    else
        %%%%�����������%%%
        %N(unique(a{2}.train.y))=length(a{2}.train.y)/length(unique(a{2}.train.y));
        N=[N,length(loc1)];
        new_C=[new_C,mean(a.train.X(:,loc1),2)];
        new_Ytr=[new_Ytr,y(i)];
    end
end
c=length(N);

%%%%%%%%%%%%%%%%%%%%%%%%����K_C%%%%%%%%%%%%%%%%%%%%%%%%%%%%
new_K_C=zeros(c,c);
%%%%�������е�%%%
new_K_C(1:old_c,1:old_c)=K_C;
for i=1:size(y,2)
    loc1=find(a.train.y==y(i));
    if ismember(y(i),Ytr)
        loc=find(Ytr==y(i));
        for j=1:old_c
            new_K_C(j,loc)=Gaussian(new_C(:,j),new_C(:,loc));
            new_K_C(loc,j)=new_K_C(j,loc);
        end
    else
        %%%%�����������%%%
        loc=find(new_Ytr==y(i));
        for j=1:c
            new_K_C(j,loc)=Gaussian(new_C(:,j),new_C(:,loc));
            new_K_C(loc,j)=new_K_C(j,loc);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
new_K_C_inv=inv(new_K_C);
end

