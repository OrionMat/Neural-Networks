function [Theta1, Theta2, Theta3, Jpast] = trainNN (X, y, lambda, numLabels);
  
  [m, n] = size(X);
  X = [ones(m, 1) X];
  Y = eye(numLabels)(y, :)';
  
  N1 = n;
  N2 = 200;
  N3 = 50;
  N4 = numLabels;
  Theta1 = randInitialise(N2, N1+1);
  Theta2 = randInitialise(N3, N2+1); 
  Theta3 = randInitialise(N4, N3+1);
  ThetaALL = [Theta1(:) ; Theta2(:) ; Theta3(:)];
  
  % compute initial cost and gradient of cost
  [J, gradient] = costFunctionNN(X, Y, ThetaALL, lambda, m, N1, N2, N3, N4);
  
  options = optimset('MaxIter', 40);
  costFunction = @(t)costFunctionNN(X, Y, t, lambda, m, N1, N2, N3, N4);
  [W, Jpast] = fmincg(costFunction, ThetaALL, options);
  
  % reshape unrolled weights back to layer weight matricies
  num_weights1 = N2*(N1+1);
  num_weights2 = N3*(N2+1);
  num_weights3 = N4*(N3+1);
  Theta1 = reshape(W(1:num_weights1), N2, N1+1); 
  Theta2 = reshape(W(num_weights1+1:num_weights1+num_weights2), N3, N2+1);
  Theta3 = reshape(W(num_weights1+num_weights2+1:end), N4, N3+1);
  
endfunction
