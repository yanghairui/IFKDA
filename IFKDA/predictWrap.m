function [precision, t_p, predictLabel, probability] = predictWrap(trainData, trainLabel, testData, testLabel)

tic
% model = fitcknn(trainData, trainLabel, 'NumNeighbors', 1, 'NSMethod', 'exhaustive', 'Distance', 'minkowski', 'Standardize', 1);

%   KNN=FITCKNN(X,Y) is an alternative syntax that accepts X as an
%   N-by-P matrix of predictors with one row per observation and one column
%   per predictor. Y is the response and is an array of N class labels. 
model = fitcknn(trainData, trainLabel, 'NumNeighbors', 1, 'Standardize', 1);

[predictLabel, probability, ~] = predict(model, testData);
precision = double(sum(predictLabel == testLabel')) / length(testLabel);

t_p=toc;

end
