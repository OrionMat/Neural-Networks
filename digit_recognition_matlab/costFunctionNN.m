function [J, gradient] = costFunctionNN (X, Y, ThetaALL, lambda, m, N1, N2, N3, N4)
  
  % reshape unrolled thetaALL weights back to seperate layer weight matricies
  num_weights1 = N2*(N1+1);
  num_weights2 = N3*(N2+1);
  num_weights3 = N4*(N3+1);
  Theta1 = reshape(ThetaALL(1:num_weights1), N2, N1+1); 
  Theta2 = reshape(ThetaALL(num_weights1+1:num_weights1+num_weights2), N3, N2+1);
  Theta3 = reshape(ThetaALL(num_weights1+num_weights2+1:end), N4, N3+1);
  
  % initialise cost and gradient and compute regularization terms
  J = 0;
  reg = (lambda/(2*m))*sum(ThetaALL.^2);
  reg_grad1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
  reg_grad2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
  reg_grad3 = (lambda/m)*[zeros(size(Theta3, 1), 1) Theta3(:, 2:end)];
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  Theta3_grad = zeros(size(Theta3));
  
  % loop though training examples accumalating cost functions value and gradient
  for i=1:m
    
    % forward propergation
    a1 = X(i, :)';
    a2 = [ones(1) ; sigmoid(Theta1*a1)];
    a3 = [ones(1) ; sigmoid(Theta2*a2)];
    a4 = sigmoid(Theta3*a3);
    hx = a4;  
    
    % accumulate examples cost
    y = Y(:, i);
    J = J + sum(-y.*log(hx)-(1-y).*log(1-hx));
    
    % backpropagation (compute cost gradient)
    d4 = hx - y;
    d3 = (Theta3'*d4).*a3.*(1-a3);
    d2 = (Theta2'*d3(2:end)).*a2.*(1-a2);
    Theta1_grad = Theta1_grad + d2(2:end)*a1';
    Theta2_grad = Theta2_grad + d3(2:end)*a2';
    Theta3_grad = Theta3_grad + d4*a3';
    
  end
  
  Theta1_grad = (1/m)*Theta1_grad + reg_grad1;
  Theta2_grad = (1/m)*Theta2_grad + reg_grad2;
  Theta3_grad = (1/m)*Theta3_grad + reg_grad3;
  
  J = J/m + reg;
  gradient = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];
  
endfunction
