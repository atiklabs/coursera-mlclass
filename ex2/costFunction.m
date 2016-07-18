function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% === Option 1: Iterative ===
% for i = 1:m
%   s = sigmoid(X(i, :)*theta);
%   J += y(i)*log(s) + (1 - y(i))*log(1 - s);
% end
% J = -J/m;

% === Option 2: Vectorized ===
h = sigmoid(X*theta);
J = -sum(y.*log(h) + (1 - y).*log(1 - h))/m;

% === Option 1: Iterative ===
% for j = 1:size(grad)
%   for i = 1:m
%     grad(j) += (sigmoid(X(i, :)*theta) - y(i))*X(i, j);
%   end
%   grad(j) = 1/m*grad(j);
% end

% === Option 2: Vectorized ===
grad = sum((h - y).*X)./m;

% =============================================================

end
