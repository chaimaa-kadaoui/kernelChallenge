addpath(genpath(pwd))

%% Load Data

x_test = csvread('Xte.csv');
x_test = x_test(:,1:end-1);
x_train_total = csvread('Xtr.csv');
x_train_total = x_train_total(:,1:end-1);
y_train_total = csvread('Ytr.csv');

%Moving input data to grayscale
x_train_total = reshape(x_train_total, [5000,1024,3]);
x_train_total = mean(x_train_total,3);

%Moving input data to grayscale
x_test = reshape(x_test, [2000,1024,3]);
x_test = mean(x_test,3);

% Preprocess data

%x_train_total_p = x_train_total;
%x_test_p = x_test;

window_size = 8;
stride = 8;

x_train_total_p = processHOG(x_train_total,window_size,stride,2);
x_test_p = processHOG(x_test,window_size,stride,2);

%% Train SVM

% train one-against-one models
numLabels = length(unique(y_train_total(:,2)));
model = cell(numLabels,numLabels);
for k=1:numLabels
    for l=k+1:numLabels
        fprintf('Computing SVM for class %i vs %i\n',k-1,l-1);
        x_train_partial = x_train_total_p((y_train_total(:,2) == k-1) | (y_train_total(:,2) == l-1),:);
        y_train_partial = y_train_total((y_train_total(:,2) == k-1) | (y_train_total(:,2) == l-1),:);
        model{k,l} = fitSVMPosterior(fitcsvm(x_train_partial, double(y_train_partial(:,2)==k-1),'KernelFunction','RBF'));
    end
end

%% Get the posterior probability matrix for the predictions

% get probability estimates of test instances using each model
numTest = size(x_test,1);
prob = zeros(numTest,numLabels,numLabels);
for k=1:numLabels
    for l=k+1:numLabels
        fprintf('Computing posteriors for class %i vs %i\n',k-1,l-1);
        [~,p] = predict(model{k,l}, x_test_p);
        prob(:,k,l) = p(:,model{k,l}.ClassNames==1) > 0.5;    % decision of class==k-1 vs class==l-1
        prob(:,l,k) = p(:,model{k,l}.ClassNames==1) <= 0.5;   %decision for class==l-1 vs class==k-1
    end
end
%% Do the prediction

% predict the class with the max vote
[~,pred] = max(sum(prob,3),[],2);
pred = pred-1;
pred = [(1:numTest)' pred];

% write prediction to file
path = './results/Yte.csv';
csvfile = fopen(path,'w');
fprintf(csvfile,'Id,Prediction\n');
fclose(csvfile);
dlmwrite (path, pred, '-append');


