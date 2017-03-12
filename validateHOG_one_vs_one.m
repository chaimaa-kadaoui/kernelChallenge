function [acc, mean_acc] = validateHOG_one_vs_one(x_train_total,y_train_total, window_size, stride, kernel, p_norm)

    %% Preprocess data with windowed HOG

    x_train_total_p = processHOG(x_train_total,window_size,stride, p_norm);

    %% Preprocess data with Harris corner detector based HOG
%     sigma_derivation = 1;
%     sigma_integration = 2;
%     epsilon_harris = 0.0001;
%     derivator_size = 9;
%     number_of_histo = 40;
%     c_anms = 0.99;
%     x_train_total_p = zeros(size(x_train_total,1),12*number_of_histo);
%     parfor i = 1:size(x_train_total,1)
%         % Harris corner detector
%         img_harris = harris_corner_detector(reshape(x_train_total(i,:),[32,32]),sigma_derivation,sigma_integration,epsilon_harris,derivator_size);
%         % ANMS
%         points_of_interest = anms(img_harris,number_of_histo,c_anms);
%         %HOG with centers
%         x_p = processHOGrad(x_train_total(i,:), window_size, points_of_interest, p_norm);
%         x_train_total_p(i,:) = x_p;
%         if mod(i,100) == 0
%           fprintf('Computed image : %i\n',i);
%         end
%     end
    
    %% Cross-validation

    splitPart = 10;
    splitSize = floor(size(x_train_total,1)/splitPart);

    acc = zeros(1,splitPart);

    for valPart=1:splitPart
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
                x_train_partial = x_train((y_train(:,2) == k-1) | (y_train(:,2) == l-1),:);
                y_train_partial = y_train((y_train(:,2) == k-1) | (y_train(:,2) == l-1),:);
                model{k,l} = fitSVMPosterior(fitcsvm(x_train_partial, double(y_train_partial(:,2)==k-1),'KernelFunction','RBF'));
            end
        end

        %% Get the posterior probability matrix for the predictions

        % get probability estimates of test instances using each model
        numTest = size(x_val,1);
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
        fprintf('Accuracy using the validation split %i: %f\n',valPart, acc(valPart));
        if acc<0.4
            fprintf('Validation stopped in advance\n')
            break
        end
    end

    mean_acc = mean(acc(acc>0));
    fprintf('Average accuracy with kernel %s window_size %i and stride %i: %f\n', kernel, window_size, stride, mean_acc);
end