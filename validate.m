%% Preprocess data

x_train_total_p = x_train_total;

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

    % train one-against-all models
    numLabels = length(unique(y_train(:,2)));
    model = cell(numLabels,1);
    for k=1:numLabels
        fprintf('Computing SVM for class %i using the validation split %i\n',k, valPart);
        model{k} = fitSVMPosterior(fitcsvm(x_train_p, double(y_train(:,2)==k-1),'KernelFunction','RBF'));
    end

    %% Get the posterior probability matrix for the predictions

    % get probability estimates of test instances using each model
    numTest = size(x_val,1);
    prob = zeros(numTest,numLabels);
    for k=1:numLabels
        fprintf('Computing posteriors for class %i using the validation split %i\n',k);
        [~,p] = predict(model{k}, x_val_p);
        prob(:,k) = p(:,model{k}.ClassNames==1);    % probability of class==k
    end

    % predict the class with the highest probability
    [~,pred] = max(prob,[],2);
    pred = pred-1;
    acc(valPart) = sum(pred == y_val(:,2))./ numTest;         % accuracy
    fprintf('Accuracy using the validation split %i: %f\n',valPart, acc(valPart));
end

mean_acc = mean(acc);
fprintf('Average accuracy: %f\n', mean_acc);