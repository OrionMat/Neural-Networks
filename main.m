clc, clear, close all

load('handDigits.mat');
numLabels = 10;

% split into trainig, validation and testing sets
mAll = length(y);
train_split = 0.6;
val_split = 0.2;
[Xtrain, ytrain, Xval, yval, Xtest, ytest] = testTrainValSplit ([X y], mAll, train_split, val_split);

% train neural network
lambda = 1;
[Theta1, Theta2, Theta3, Jpast] = trainNN(Xtrain, ytrain, lambda, numLabels);

evaluation