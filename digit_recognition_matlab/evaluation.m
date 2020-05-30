% evaluate training error:
A1 = [ones(size(Xtrain, 1), 1) Xtrain]';
A2 = [ones(1, size(Xtrain, 1)) ; sigmoid(Theta1*A1)];
A3 = [ones(1, size(Xtrain, 1)) ; sigmoid(Theta2*A2)];
A4 = sigmoid(Theta3*A3);
Hx = A4;
[~, idxs] = max(Hx);
predictions = idxs';
fprintf('training set accuracy: %f\n', mean(double(predictions == ytrain)) * 100);

% validate neural network:
A1 = [ones(size(Xval, 1), 1) Xval]';
A2 = [ones(1, size(Xval, 1)) ; sigmoid(Theta1*A1)];
A3 = [ones(1, size(Xval, 1)) ; sigmoid(Theta2*A2)];
A4 = sigmoid(Theta3*A3);
Hx = A4;  
[~, idxs] = max(Hx);
predictions = idxs';
fprintf('validation set accuracy: %f\n', mean(double(predictions == yval)) * 100);

errorIdxs = find(~(predictions == yval));
valErrors = Xval(errorIdxs, :);
displayDigitGrid(valErrors(1:64, :));

valErrLabels = yval(errorIdxs, :);
valErrPredictions = predictions(errorIdxs, :);
dispErrLables = reshape(valErrLabels(1:64), 8, 8)
dispErrPrediction = reshape(valErrPredictions(1:64), 8, 8)

% plot reduction in cost
figure
plot(Jpast);
xlabel('itteration');
ylabel('cost');
title('training cost');