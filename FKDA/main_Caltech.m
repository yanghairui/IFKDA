clc;
clear;
disp('FKDA starting');

for kk=2:10
    load(sprintf('../data/Caltech256_ObjectCategories/Caltech4096_chunk_non_%d.mat',kk));

    Xtr=batch.train.X;
    ytr=batch.train.y;
    Xte=batch.test.X;
    yte=batch.test.y;
    [d,n]=size(Xtr); % the dimension and number of all samples
    Ytr=unique(ytr,'stable');
    c=length(Ytr);   % the number of all sample classes

    %%先批量%%%
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

    %%%%增量chunk%%%%
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
        %%%%读取增量的样本%%
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
    % save(sprintf('AWA/chunk/vgg19_chunk_%d.mat',kk),'T_sum','T_p','Pre');
    % save(sprintf('Caltech/chunk/4096_200chunk_%d.mat',kk),'T_sum','T_p','Pre');
    %clear
    %clc
    disp(Pre);
end

disp(mean(Pre(:, :), 2))