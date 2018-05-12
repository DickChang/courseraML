function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
h = sigmoid( X * theta );
for iter = 1:m
    yi = y(iter);
    hi = h(iter);
    J = J + ( -(yi*log(hi)) ) - ( (1-yi)*log(1-hi) );
end
%J = J / (m-1);
%J = J + ( (lambda/2/(m-1))*sum(theta(2:end).**2) );
J = J / m;
J = J + ( (lambda/2/m)*sum(theta(2:end).**2) );


hError = h - y;
grad(1) = sum( (hError) .* X(:,1) ) / m;
%lm = lambda/(m-1);
lm = lambda/m;
for iter = 2:length(theta)
    %grad(iter) = sum( (hError) .* X(:,iter) ) / (m-1);
    grad(iter) = sum( (hError) .* X(:,iter) ) / m;
    grad(iter) = grad(iter) + (lm * theta(iter));
end

% =============================================================

end
