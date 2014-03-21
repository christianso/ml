function p = min_callback(ii, nn_params, cost, X, y, nn, lambda, X_cv, y_cv)
	
	global stat_data;
	if(ii==1)
		stat_data = [];
	end

	Theta1 = reshape(nn_params(1:nn(2) * (nn(1) + 1)), nn(2), (nn(1) + 1));
	Theta2 = reshape(nn_params((1 + (nn(2) * (nn(1) + 1))):end), nn(3), (nn(2) + 1));
	
	[p,n] = checkmodel(X,y,Theta1, Theta2, false);

	if size(X_cv > 0) 
		[p_cv,n_cv] = checkmodel(X_cv,y_cv,Theta1, Theta2, false);
		cost_cv = nnCostFunction(nn_params, nn(1), nn(2), nn(3), X_cv, y_cv, lambda);
		stat_data = [stat_data; ii cost cost_cv]; 
		fprintf('Iteration %4i | %2.3f%% %2.4f | %2.3f%% %2.4f| \n', ii, 100*p/(p+n), cost, 100*p_cv/(p_cv+n_cv), cost_cv);
	else 
		fprintf('Iteration %4i | %2.2f%% %2.4f | \n', ii, 100*p/(p+n), cost);
	end
	%fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
end