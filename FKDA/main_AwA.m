%%%%%chunk%%%%
clear
clc;
for kk=1:10
load(sprintf('../data/AWA_noveltydetection1/AWA_non_%d.mat',kk));
    Xtr=batch.train.X;
    ytr=batch.train.y;
    Xte=batch.test.X;
    yte=batch.test.y;
    [d,n]=size(Xtr); % the dimension and number of all samples
    Ytr=unique(ytr,'stable');
    c=length(Ytr);   % the number of all sample classes
        
    tic
    [C,N]=KFDA(Xtr,ytr);
    K_C=KGaussian(C);
    K_C_inv=inv(K_C);
    K_cz=Gaussian(C,Xte);
    t_K=toc;
    tic
    P=K_C_inv*K_cz;
    P_c=eye(c);
    t_batch=toc;
    [predictLabel, precision,t_p,probability]=predictWrap(P_c',Ytr,P',yte);
    
    new_X=Xtr;
    new_XLabel=ytr;
    new_C=C;
    new_K_C=K_C;
    T_p=[];
    T_p=[T_p;t_p];
    T_sum=[];
    T_sum=[T_sum;t_K];
    T_sum=[T_sum;t_batch];
    Pre=[];
    Pre=[Pre;precision];
    for i=1:size(Inc,2)
        z=Inc{i};
        alable=unique(z.train.y,'stable');
        Xte=[Xte,z.test.X];
        yte=[yte,z.test.y];
        tic
        [new_K_C,new_K_C_inv,N,new_C,Ytr,c]=Inc_KFDA(new_K_C,N,z,new_C,Ytr);
        K_cz=ch_Gaussian(new_C,Xte,K_cz,alable,Ytr);
        P=new_K_C_inv*K_cz;
        P_c=eye(c);
        sum_t=toc;
        [predictLabel, precision,t_p,probability]=predictWrap(P_c',Ytr,P',yte);
        T_sum=[T_sum;sum_t];
        T_p=[T_p;t_p];
        Pre=[Pre;precision];
    end
    disp(Pre);
end

function [C,N,CLabel,c] = KFDA( X,XLabel )
 [d,n]=size(X);
CLabel=unique(XLabel,'stable');
c=length(CLabel);  % the number of all sample classes
N=zeros(1,c);
%%%%%%compute M
C=zeros(d,c);
k=1;
for i=1:c
    loc=[];
    loc=find(XLabel==CLabel(i));
    N(k)=length(loc);
    C(:,k)=mean(X(:,loc),2);
    k=k+1;
end
end

function K = KGaussian(A)
[gamma,r] = size(A);
K1=zeros(r,r);
for i = 1: r
    for j = i: r
        dis=A(:,i)-A(:,j);
        K1(i,j) = exp( -(   norm( dis)^2/ (gamma) ));
    end
end
 
K=K1'+K1-eye(r);
end

function [new_K_C,new_K_C_inv,N,new_C,new_Ytr,c] = Inc_KFDA(K_C,N,a,C,Ytr)
old_N=N;
old_c=length(old_N);
new_C=C;
y=unique(a.train.y,'stable');
new_Ytr=Ytr;
for i=1:size(y,2)
    loc1=find(a.train.y==y(i));
    if ismember(y(i),Ytr)
        loc=find(Ytr==y(i));
        N(loc)=N(loc)+length(loc1);
        new_C(:,loc)=(old_N(loc)*new_C(:,loc)+sum(a.train.X(:,loc1),2))/N(loc);
    else
        N=[N,length(loc1)];
        new_C=[new_C,mean(a.train.X(:,loc1),2)];
        new_Ytr=[new_Ytr,y(i)];
    end
end
c=length(N);
 
new_K_C=zeros(c,c);
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
        loc=find(new_Ytr==y(i));
        for j=1:c
            new_K_C(j,loc)=Gaussian(new_C(:,j),new_C(:,loc));
            new_K_C(loc,j)=new_K_C(j,loc);
        end
    end
end
new_K_C_inv=inv(new_K_C);
end

function [predictLabel, precision,t_p,probability] = predictWrap(trainData, trainLabel, testData, testLabel)
tic
model = fitcknn(trainData, trainLabel, 'NumNeighbors', 1, 'Standardize', 1);
[predictLabel,probability,~] = predict(model, testData);
precision = double(sum(predictLabel == testLabel')) / length(testLabel);
t_p=toc;
end

function K= Gaussian( X,z )
 [gamma r] = size(X);
 [m mm]=size(z);
K=zeros(r,mm);
if gamma ~=m
    error('the dimension of input data is inconsistent!');
else
    for i = 1: r
        for j=1:mm
            dis=X(:,i)-z(:,j);
            K(i,j) = exp( -(   norm( dis)^2/ (gamma)) );
        end
    end
end
end

function K= ch_Gaussian(new_C,z,K_xz,alabel,Ytr )
[gamma,r] = size(new_C);
[m,mm]=size(z);%%% z is the set of test samples
K=zeros(r,mm);
[a,b]=size(K_xz);
if gamma ~=m
    error('the dimension of input data is inconsistent!');
else
    K(1:a,1:b)=K_xz;
    for ii=1:length(alabel)
        i=find(Ytr==alabel(ii));
        for j=1:b
            dis=new_C(:,i)-z(:,j);
            K(i,j) = exp(-(   norm( dis )^2 / (gamma)) );
        end
    end
    for i = 1: r
        for j=(b+1):mm
            dis=new_C(:,i)-z(:,j);
            K(i,j) = exp( -(   norm( dis )^2 / (gamma)) );
        end
    end
    
end
end


