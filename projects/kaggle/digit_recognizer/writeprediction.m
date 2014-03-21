function [positives, negatives] = writeprediction(X,T1,T2, filename)
		
	pred = predict(T1, T2, X); 
	pred(find(pred==10)) = 0;
	
	pred = [1:size(pred);pred']';
	old_val = save_header_format_string ('ImageId,Label');
	dlmwrite(filename, pred,',',0,0);
	save_header_format_string (old_val);
		
end
	
	

