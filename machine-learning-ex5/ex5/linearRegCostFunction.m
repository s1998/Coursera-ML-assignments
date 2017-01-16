function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


%disp(theta)

h_of_x = X * theta;

%fprintf('Display size of h_of_x and y and X and theta');
%disp(size(h_of_x));
%disp(size(y));
%disp(size(X));
%disp(size(theta));

h_of_x = h_of_x-y;


J = (1 / (2*m)) * (sum(h_of_x' * h_of_x)) + (lambda)/(2*m) * (sum(theta' * theta) - theta(1)^2);

p_of_x = (h_of_x' * X)';
grad =  (1 / (m)) * (p_of_x) + (lambda)/(m) * [ 0; theta(2:end)];

% =========================================================================

grad = grad(:);

end
