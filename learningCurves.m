clc, clear, close all

load('handDigits.mat');
numLabels = 10;

% split into trainig, validation and testing sets
mAll = length(y);
train_split = 0.6;
val_split = 0.2;
[Xtrain, ytrain, Xval, yval, Xtest, ytest] = testTrainValSplit ([X y], mAll, train_split, val_split);

% define NN architecture
N1 = size(Xtrain, 2);
N2 = 200;
N3 = 50;
N4 = numLabels;
lambda = 1;

figure
hold on
% varying size of training set
[mTrain, ~] = size(Xtrain);
[mVal, ~] = size(Xval);
for m = 1:100:mTrain
  
  trainSet = Xtrain(1:m, :);
  trainLab = ytrain(1:m);
  
  % train neural network
  [Theta1, Theta2, Theta3, ~] = trainNN(trainSet, trainLab, lambda, numLabels);
  ThetaALL = [Theta1(:) ; Theta2(:) ; Theta3(:)];
  
  % training and validation cost
  oneHotTrainLabs = eye(numLabels)(trainLab, :)';
  oneHotValLabs = eye(numLabels)(yval, :)';
  [Jtrain, ~] = costFunctionNN ([ones(m, 1) trainSet], oneHotTrainLabs, ThetaALL, 0, m, N1, N2, N3, N4)
  [Jval, ~] = costFunctionNN ([ones(mVal, 1) Xval], oneHotValLabs, ThetaALL, 0, mVal, N1, N2, N3, N4)
  plot(m, Jtrain, 'bo', 'markersize', 10);
  plot(m, Jval, 'r+', 'markersize', 10);
end

% evaluate training error:
A1 = [ones(size(trainSet, 1), 1) trainSet]';
A2 = [ones(1, size(trainSet, 1)) ; sigmoid(Theta1*A1)];
A3 = [ones(1, size(trainSet, 1)) ; sigmoid(Theta2*A2)];
A4 = sigmoid(Theta3*A3);
Hx_val = A4;
[~, idxs] = max(Hx_val);
predictions = idxs';
fprintf('training set accuracy: %f\n', mean(double(predictions == trainLab)) * 100);