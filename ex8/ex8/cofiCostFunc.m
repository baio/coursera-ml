function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%size(Theta)
%size(X)
%size(Y)
%size(R)
%size(X_grad)
%size(Theta_grad)

J = sum( sum(((X * Theta' - Y).*R).^2) ) / 2 + lambda * sum(sum(Theta .^ 2)) / 2 + lambda * sum(sum(X .^ 2)) / 2;

%X_grad = sum( ( ( (X * Theta' - Y) .* R ) * Theta ) )
%Theta_grad = sum( ( ( (X * Theta' - Y) .* R ) * X  ) )

% =============================================================
for i=1:num_movies
	%find indexes of users who rated this movie
	idx = find(R(i, :) == 1);
	%get features rate only for users who rated this movie
	thetai = Theta(idx, :);
	%get rating of movie only for users who rated it
	yi = Y(i, idx);
	
	% number of movie fetures for particular movie 
	% * (matrix of users (who rated the movie) X rate of the features for this user)
	X_grad(i,:) = (X(i,:) * thetai' - yi) * thetai + lambda * X(i, :);
endfor;	

for i=1:num_users
	%find indexes of movies which have been rated by this user
	idx = find(R(:, i) == 1);
	
	%get features rate only for movies which have been rated by this user
	xi = X(idx, :);
	
	%get rating of the movies, which have been rated by the user
	yi = Y(idx, i);
	
	% number of movie fetures for particular user 
	% * (matrix of users (who rated the movie) X rate of the features for this user)
	Theta_grad(i,:) = ( xi * Theta(i, :)' - yi )' * xi + lambda * Theta(i, :);
endfor;	


grad = [X_grad(:); Theta_grad(:)];

end
