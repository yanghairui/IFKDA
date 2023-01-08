function K = KGaussian(A)
% for building kernel data matrix, reduced or full, with Gaussian kernel.
%
% Inputs
% A: full data set.
%
% Outputs
% K: kernel data, full or reduced.

[gamma, r] = size(A); % gamma: attribute dim; r: quantity dim
K1 = zeros(r, r);     % K1: initial kernel data matrix
for i = 1 : r
    for j = i : r
        dis = A(:, i) - A(:, j); % distance between column i and column j
        K1(i, j) = exp(-(norm(dis)^2 / (gamma)));
    end
end

K = K1' + K1 - eye(r); % K is a Symmetric Matrix
end