clear;
clc;
disp('IFKDA starting');
disp('-------------------------------');

for kk = 1 : 10
    load(sprintf('../data/AWA_noveltydetection1/AWA_non_%d.mat', kk));
    
    Xtr = batch.train.X; % Xtr : train data         AWA - [4096, 2240]
    ytr = batch.train.y; % ytr : train label
    Xte = batch.test.X;  % Xte : test data
    yte = batch.test.y;  % yte : test label    
    Ytr = unique(ytr, 'stable'); % Ytr : known class label
    
    % 1. FKDA  
    [Co, No, Ko, Kcz, P_z, P_c, t_K, t_batch] = FKDA(Xtr, ytr, Xte);   
    [precision, t_p] = predictWrap(P_c', Ytr, P_z', yte);
    
    % 2. IFKDA  
    chunk_num = size(Inc, 2); 
    
    T_sum = zeros(1, chunk_num);
    T_sum(1) = t_K;
    T_sum(2) = t_batch;
    T_p = zeros(1, chunk_num);
    T_p(1) = t_p;
    Pre = zeros(1, chunk_num);
    Pre(1) = precision;
       
    for i = 1 : chunk_num
        % read chunk data
        chunk = Inc{i};
        yte = [yte, chunk.test.y];
        
        [Ko, Kcz, Co, No, Xte, P_c, Ytr, P_z, t_sum] = IFKDA(Ko, Kcz, Co, No, Ytr, chunk, Xte);
        [precision, t_p] = predictWrap(P_c', Ytr, P_z', yte);

        T_sum(2+i) = t_sum;
        T_p(1+i) = t_p;
        Pre(1+i) = precision;
    end
    %save(sprintf('AWA/chunk/vgg19_chunk_%d.mat', kk), 'T_sum', 'T_p', 'Pre');
    %save(sprintf('Caltech/chunk/4096_200chunk_%d.mat', kk), 'T_sum', 'T_p', 'Pre');
    %clear
    %clc
    disp(Pre);
end
disp('-------------------------------');
disp(mean(Pre(:, :), 2))