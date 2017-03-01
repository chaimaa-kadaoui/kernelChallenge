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

%% Cross-validation

splitPart = 10;
valPart = 1;

boarderIndex = floor((splitPart-valPart)/splitPart*size(x_train_total,1));

x_val = x_train_total(boarderIndex+1:end,:);
x_train = x_train_total(1:boarderIndex,:);

y_val = y_train_total(boarderIndex+1:end,:);
y_train = y_train_total(1:boarderIndex,:);

%% Kernel

gamma = 0.5;
gram_train = rbf(x_train,x_train,gamma);
gram_val = rbf(x_train,x_val,gamma);
C = 1;
%% Train SVM

%# train one-against-all models
numLabels = length(unique(y_train(:,2))); %Handle class 0
models = cell(numLabels,1);
for k=1:numLabels
    fprintf('Computing SVM for class %i\n',k);
    y_bin = zeros(size(y_train,1),1);
    y_bin(y_train(:,2)==k-1)=1;
    y_bin(y_train(:,2)~=k-1)=-1;
    [alpha_y, bias] = fitcsvm_kernel(gram_train, y_bin, C);
    models{k}.alpha_y = alpha_y;
    models{k}.bias = bias;
    models{k}.post_proba = fit_posterior(gram_train, alpha_y, bias);
end

%% Get the posterior probability matrix for the predictions

%# get probability estimates of test instances using each model
numTest = size(x_val,1);
prob = zeros(numTest,numLabels);
for k=1:numLabels
    fprintf('Computing SVM for class %i\n',k);
    post_proba = fit_posterior(gram_val, models{k}.alpha_y, models{k}.bias);
    prob(:,k) = post_proba(:,1);    %# probability of class==k
end

%# predict the class with the highest probability
[~,pred] = max(prob,[],2);
pred = pred - 1;
acc = sum(pred == y_val) ./ size(y_val,1);    %# accuracy
fprintf('Resulting accuracy : %f\n',acc);

