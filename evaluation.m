% evaluate training error:
A1 = [ones(size(Xtrain, 1), 1) Xtrain]';
A2 = [ones(1, size(Xtrain, 1)) ; sigmoid(Theta1*A1)];
A3 = [ones(1, size(Xtrain, 1)) ; sigmoid(Theta2*A2)];
A4 = sigmoid(Theta3*A3);
Hx = A4;
[~, idxs] = max(Hx);
fprintf('training set accuracy: %f\n', mean(double(idxs' == ytrain)) * 100);

% validate neural network:
A1 = [ones(size(Xval, 1), 1) Xval]';
A2 = [ones(1, size(Xval, 1)) ; sigmoid(Theta1*A1)];
A3 = [ones(1, size(Xval, 1)) ; sigmoid(Theta2*A2)];
A4 = sigmoid(Theta3*A3);
Hx = A4;  
[~, idxs] = max(Hx);
fprintf('validation set accuracy: %f\n', mean(double(idxs' == yval)) * 100);

errorIdxs = find(~(idxs' == yval));
valErrors = (Xval(errorIdxs, :));
displayDigitGrid(valErrors(1:64, :));