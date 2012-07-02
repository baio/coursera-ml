function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    derivative1 = computeDerivative(X, y, theta, ones(m,2))
    derivative2 = computeDerivative(X, y, theta, X)

    theta(1,1) = theta(1,1) - alpha * derivative1
    theta(2,1) = theta(2,1) - alpha * derivative2

    %must be less than on pervious step
    %computeCost(X, y, theta)
    %grdient
    %alpha * derivative1
    %alpha * derivative2

    % ============================================================



    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end


function D = computeDerivative(X, y, theta, x)
m = length(y);
D = 0;
for iter = 1:m
    D = D + (theta' * X(iter,:)' - y(iter,:))' * x(iter,:);
end

D = D / m;

end

