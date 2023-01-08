function [Ko, Kcz, Co, No, Xte, P_c, Ytr, P_z, t_sum] = IFKDA(Ko, Kcz, Co, No, Ytr, chunk, Xte)
tic

Xtr1 = chunk.train.X;
ytr1 = chunk.train.y;
Xte1 = chunk.test.X;
Ytr1 = unique(ytr1, 'stable'); % Ytr1 : chunk class label

% projection of new samples in k dimension space
Kcz1 = Gaussian(Co, Xte1); % compute <phi(ci), new_z> -- add column
Kcz = [Kcz, Kcz1];
Xte = [Xte, Xte1];               % add novel class data into test data

% 1. update Co as FLDA
class_num = size(Ytr1, 2);     % class_num : chunk class number
for i = 1 : class_num    
    ind = find(ytr1 == Ytr1(i)); % ind: samples of class i in a chunk
    % process samples of known class
    if ismember(Ytr1(i), Ytr)
        % update Co of known class
        ki = find(Ytr == Ytr1(i));
        No(ki) = No(ki) + length(ind);
        Co(:, ki) = (No(ki) * Co(:, ki) + sum(Xtr1(:, ind), 2)) / No(ki);
        
        % update Ko by known class
        for j = 1 : length(Ytr)
            Ko(j, ki) = Gaussian(Co(:, j), Co(:, ki)); % update a column
            Ko(ki, j) = Ko(j, ki);                     % update a row
        end
        
        % update Kcz by known class
        for zi = 1 : size(Xte, 2)
            Kcz(ki, zi) = Gaussian(Co(:, ki), Xte(:, zi));
        end
    else
        % process samples of known class
        No = [No, length(ind)];
        
        % update center of novel class
        Cn = mean(Xtr1(:, ind), 2);
        Co = [Co, Cn];
        
        % update Ko by novel class
        k = length(Ytr);
        Ytr = [Ytr, Ytr1(i)];        
        Ko = [     Ko,      zeros(k, 1); ...        % add a column
               zeros(1, k),      0       ];         % add a row
        for j = 1 : k+1
            Ko(j, k+1) = Gaussian(Co(:, j), Co(:, k+1)); % update a column
            Ko(k+1, j) = Ko(j, k+1);                     % update a row
        end
        
        % update Kcz by novel class
        Kcz = [Kcz; Gaussian(Cn, Xte)];
    end
end
k = length(Ytr);

% 2. compute projection of sample z
P_z = Ko \ Kcz;
P_c = eye(k);

t_sum = toc;
end

