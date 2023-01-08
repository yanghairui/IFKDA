function [Co, No, Ko, Kcz, P_z, P_c, t_K, t_batch] = FKDA(Xtr, ytr, Xte)
% 
% Inputs
% Xtr: train samples.
% ytr: train labels.
%
% Outputs
% Co   [d, k] : center matrix;
% No   [1, k] : sample number of each class
%

% 1. compute Co
Ytr = unique(ytr, 'stable'); % classes
k = length(Ytr);      % class number
No = zeros(1, k);     
d = size(Xtr, 1);     % d: feature dimension
Co = zeros(d, k);     
for i = 1 : k
    loc = find(ytr == Ytr(i)); % loc: samples of i_th class
    No(i) = length(loc);            
    Co(:, i) = mean(Xtr(:, loc), 2);% C(:, i): center of i_th class
end

% 2. compute Ko and Kcz
tic
Ko = KGaussian(Co);      % K_C  : kernel matrix
Kcz = Gaussian(Co, Xte); % K_cz : kernel vector
t_K = toc;

% 3. compute projection of sample z
tic
P_z = Ko \ Kcz;          % P_z  : coordinate of sample 'z'
P_c = eye(k);            % P_c  : coordinate of class center
t_batch = toc;

end

