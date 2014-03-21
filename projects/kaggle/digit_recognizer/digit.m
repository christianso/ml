%% Initialization
clear ; close all; clc
addpath('~/octave/ml/functions/');

%% =========== Load training data =============

load 'data/data';

X = train(2:end,2:end);
y = train(2:end,1);

y(find(y==0)) = 10;


%% =========== Randomize data =============

%randix = randperm(size(labels));
%images = images(randix,:);
%labels = labels(randix);

%% =========== split into train and xval=============

train_count = int32(size(y)(1)*0.80);
X_cv = X(train_count+1:end,:);
y_cv = y(train_count+1:end,:);
X = X(1:train_count,:);
y = y(1:train_count,:);


%% =========== Declaring work variables =============
input_layer_size  = size(X)(2);  % width * height

num_labels = max(y) ;
m = size(X, 1);

%% =========== Visualizing image data =============
%% Visualize random sample of all

%sel = randperm(size(X, 1));
%sel = sel(1:64);
%displayData(X(sel, :),sqrt(input_layer_size));

%% Visualize random sample of certain label logos

%logoY = find(y==1);
%sel = randperm(size(logoY));
%sel = sel(1:49);
%displayData(X(logoY(sel),:));



%% ================ Part 2: Loading Parameters ================

%Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
%Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
  
%nn_params = [Theta1(:) ; Theta2(:)];

%% ================ Part 3: Compute Cost (Feedforward) ================

%lambda = 0;
%J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
%fprintf(['Cost at random parameters with lambda %d: %f \n(this value should be about 0.287629)\n'], lambda, J);

				
%% =============== Part 4: Implement Regularization ===============	
			
%lambda = 1;
%J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
%fprintf(['Cost at random parameters with lambda %d: %f \n(this value should be about 0.287629)\n'], lambda, J);
						
%% ================ Part 5: Sigmoid Gradient  ================												    

%g = sigmoidGradient([1 -0.5 0 0.5 1]);
%fprintf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]: \n');
%fprintf('%f',g);
%fprintf('\n\n');

%% ================ Part 6: Initializing Pameters ================

global stat_data;
epsilon = [0.12];
lambda =  [1];
hidden_layer_size = [525,526,527,528,529,530,531,532,533,534,535];
iterations = [112];

fprintf('\nTrain %d CV %d',size(X)(1), size(y_cv)(1));
for j = 1:size(epsilon)
	for l = 1:size(hidden_layer_size)
		initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size(l), epsilon(j));
		initial_Theta2 = randInitializeWeights(hidden_layer_size(l), num_labels, epsilon(j));
		hash = sum(sum(initial_Theta1))+sum(sum(initial_Theta2));
		for i = 1:size(iterations)
			for k = 1:size(lambda)
				fprintf('\nhash:%f it:%d eps:%f lam:%f neurons:%d\n',hash, iterations(i),epsilon(j),lambda(k),hidden_layer_size(l));
				Theta1 = initial_Theta1;
				Theta2 = initial_Theta2;
				initial_nn_params = [Theta1(:) ; Theta2(:)];
				
				
				callback = @(p,q,r) min_callback(p, q, r, X, y, [input_layer_size hidden_layer_size(l) num_labels],lambda(k), X_cv, y_cv);
				costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size(l), num_labels, X, y, lambda(k));
				options = optimset('MaxIter', iterations(i), 'callback', callback);
							
				[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
				Theta1 = reshape(nn_params(1:hidden_layer_size(l) * (input_layer_size + 1)), hidden_layer_size(l), (input_layer_size + 1));
				Theta2 = reshape(nn_params((1 + (hidden_layer_size(l) * (input_layer_size + 1))):end), num_labels, (hidden_layer_size(l) + 1));
				
				cost = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size(l), num_labels, X, y, lambda(k));
				plot(stat_data(:,2:end));
				writeprediction(test(2:end,:), Theta1, Theta2, 'output/predict_test_2.csv');
				
				%% ================= Part 9: Visualize Weights =================
				%%displayData(Theta1(:, 2:end));
			
				%% ================= Part 10: Implement Predict =================
				Jcv = nnCostFunction(nn_params, input_layer_size, hidden_layer_size(l), num_labels, X_cv, y_cv, lambda(k));
				fprintf('\nTrain cost: %f\n',cost);
				checkmodel(X,y,Theta1, Theta2, true);
				fprintf('\nCv cost: %f\n', Jcv);
				Jcv = nnCostFunction(nn_params, input_layer_size, hidden_layer_size(l), num_labels, X_cv, y_cv, lambda(k));
				checkmodel(X_cv, y_cv, Theta1, Theta2, true);
				
			end
		end
	end
end







