function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

cs = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmas = [0.01 0.03 0.1 0.3 1 3 10 30];
minerr = 1000;
size(cs,2)
size(sigmas,2)
for i=1:size(cs,2)
	for k = 1:size(sigmas,2)
		c = cs(i)
		s = sigmas(k)
		model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s)); 
		predictions = svmPredict(model, Xval);
		err = mean(double(predictions ~= yval))
		if minerr > err
			minerr = err
			C = c;
			sigma = s;
		endif
	endfor
endfor
				
% =========================================================================

end
