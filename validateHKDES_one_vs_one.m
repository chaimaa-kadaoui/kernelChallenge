function [acc, mean_acc] = validateHKDES_one_vs_one(x_train_total_p, y_train_total, param)

    %% Cross-validation

    splitPart = 10;
    splitSize = floor(size(x_train_total_p,1)/splitPart);

    acc = zeros(1,splitPart);

    for valPart=[1,3,6,9]
        x_train = x_train_total_p([1:(valPart-1)*splitSize, valPart*splitSize+1:end],:);
        x_val = x_train_total_p((valPart-1)*splitSize+1:valPart*splitSize,:);

        y_train = y_train_total([1:(valPart-1)*splitSize, valPart*splitSize+1:end],:);
        y_val = y_train_total((valPart-1)*splitSize+1:valPart*splitSize,:);

        %% Train SVM

		% train one-against-one models
		numLabels = length(unique(y_train_total(:,2)));
		model = cell(numLabels,numLabels);
		for k=1:numLabels
		    for l=k+1:numLabels
		        fprintf('Computing SVM for class %i vs %i\n',k-1,l-1);
				selection = (y_train_total(:,2) == k-1) | (y_train_total(:,2) == l-1);
				index = (1:size(y_train_total,1));
				index = index(selection);
				x_train_partial = x_train(index,:);
		        y_train_partial = y_train_total(index,:);
		        model{k,l} = fitSVMPosterior(fitcsvm(x_train_partial, double(y_train_partial(:,2)==k-1),'KernelFunction','linear'));
		    end
		end

        %% Get the posterior probability matrix for the predictions

        % get probability estimates of test instances using each model

		numTest = size(x_test_cut,1);
		prob = zeros(numTest,numLabels,numLabels);
		for k=1:numLabels
		    for l=k+1:numLabels
		        fprintf('Computing posteriors for class %i vs %i\n',k-1,l-1);
		        [~,p] = predict(model{k,l}, x_val);
		        prob(:,k,l) = p(:,model{k,l}.ClassNames==1) > 0.5;    % decision of class==k-1 vs class==l-1
		        prob(:,l,k) = p(:,model{k,l}.ClassNames==1) <= 0.5;   %decision for class==l-1 vs class==k-1
		    end
		end

		% predict the class with the max vote
		[~,pred] = max(sum(prob,3),[],2);
		pred = pred-1;
        acc(valPart) = sum(pred == y_val(:,2))./ numTest;         % accuracy
        fprintf('Accuracy using the validation split %i with parameters [%i,%i,%i]: %f\n',valPart,param(1),param(2),param(3),acc(valPart));
        if acc<0.5
            fprintf('Validation stopped in advance\n')
            break
        end
    end

    mean_acc = mean(acc(acc>0));
    fprintf('Average accuracy with parameters %i,%i,%i: %f\n',param(1),param(2),param(3),mean_acc);
end
