function [positives, negatives] = checkmodel(X,y,T1,T2, verbose)
	
	pred = predict(T1, T2, X); 
	
	positives_ix = find(pred==y);
	negatives_ix = find(pred!=y);
	positives = size(positives_ix)(1);
	negatives = size(negatives_ix)(1);
	if verbose
		fprintf('\nPositives:%d Negatives:%d %f\n', positives, negatives, positives/(negatives+positives));
	end
end
	
	


