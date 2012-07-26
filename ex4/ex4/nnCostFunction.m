function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%size(X)

%J = -1 * (y' * log(sigmoid(X * theta)) + (1 - y)' * log(1 - sigmoid(X * theta))) / m;

%J2 = -1 * (y * log(sigmoid(X * Theta2_grad)) + (1 -y) * log(1 - sigmoid(X * Theta2_grad))) / m;

%J = J1 + J2

a1 = [ones(m,1) X];

size(a1);
size(Theta1);

z2 = a1 * Theta1';

a2 = sigmoid(z2);

a2 = [ones(size(a2), 1) a2];

z3 = a2 * Theta2';

a3 = sigmoid(z3);

h = a3;

%%%% for first row from training set 0 (index 10)%%%%%%%%%

%yu = zeros(num_labels, 1);

%yu(y(1)) = 1;

%this number is 0 (10 index)
%y(1)

%for 0 the greatest probability must be in last column
%h(1,:)

%calculate error for positive result (y = 1), min error result should be 10
%-log(h(1,:))

%we are interested only in error for hipotese which corresponds right result
%positiveError = -log(h(1,:)) * yu

%calculate error for negative result (y = 0), max error result must be for 10
%we say wrong - and hipotese must approve this, probability = h = g(z) => ~0, (only for right result (10) should pinalize significantly but we exclude it by *)
%this way we add to the error very little (the less the better) value (if all ok)
%-log(1 - h(1,:))

%error must be insignificant
%negativeError = -log(1 - h(1,:)) * (1 - yu)

%total_step_error = positiveError + negativeError

%%%%%%%%%%%%%%now with loop%%%%%%%%%%%%

J = 0;

for i=1:m

total_step_error = 0;

yu = zeros(num_labels, 1);

yu(y(i)) = 1;

%this is result from training set
%y(i)

%max probability for the column y(i) (result)
%h(i,:)

%calculate error for positive result (y = 1), min error result should be in y(i)
%-log(h(i,:))

%we are interested only in error for hipotese which corresponds right result
positiveError = -log(h(i,:)) * yu;

%calculate error for negative result (y = 0), max error result must be for y(i)
%we say wrong - and hipotese must approve this, probability = h = g(z) => ~0, (only for right result (y(i)) should pinalize significantly but we exclude it by *)
%this way we add to the error very little (the less the better) value (if all ok)
%-log(1 - h(i,:))

%error must be insignificant
negativeError = -log(1 - h(i,:)) * (1 - yu);

total_step_error = positiveError + negativeError;

J = J + total_step_error;

endfor;

J = J / m;

%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% evctorized form %%%%%%%%%

%yunroll = repmat(1:num_labels,m,1)==repmat(y,1,num_labels);

%J = sum(sum(-1 * yunroll .* log(h) - (1 - yunroll) .* log(1 - h))) / m;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
