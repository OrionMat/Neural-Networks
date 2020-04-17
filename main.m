clc, clear, close all

load('handDigits.mat');

% split into trainig, validation and testing sets
mAll = length(y);
train_split = 0.6;
val_split = 0.2;
[Xtrain, ytrain, Xval, yval, Xtest, ytest] = testTrainValSplit ([X y], mAll, train_split, val_split);

% train neural network
numLabels = 10;
lambda = 1;
[Theta1, Theta2, Theta3, Jpast] = trainNN(Xtrain, ytrain, lambda, numLabels);

##for i=1:m
##  s = input('Paused - press enter for another example, q to exit:','s');
##  if s == 'q'
##    break
##  end    
##end  

##% display random digit
##randIdx = round(mAll*rand(1));
##displayDigit(X(randIdx, :));
##% display grid
##displayDigitGrid(X(1:100, :));
##pause;