function K = Gaussian(X, z)

[gamma, k] = size(X);   % X: train data
[d, n] = size(z);       % z: test samples
K = zeros(k, n);

if gamma ~= d
    error('the dimension of input data is inconsistent!');
else
    for i = 1: k
        for j = 1 : n
            dis = X(:, i) - z(:, j);
            K(i, j) = exp(-(norm(dis)^2 / (gamma)));
        end
    end
end

end