function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y);

% return the following variables correctly  
J = 0; 
grad = zeros(size(theta)); 

hypothesis = sigmoid(X*theta);
 

J = (-1/m) * sum( y .* log(hypothesis) + (1 - y) .* log(1 - hypothesis) );

for i = 1:m
	grad = grad + ( hypothesis(i) - y(i) ) * X(i, :)';
end
grad = (1/m) * grad;
end
