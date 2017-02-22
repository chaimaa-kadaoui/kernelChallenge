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

%% Train SVM

%# train one-against-all models
numLabels = length(unique(y_train(:,2))); %Handle class 0
model = cell(numLabels,1);
for k=1:numLabels
    fprintf('Computing SVM for class %i\n',k);
    model{k} = fitSVMPosterior(fitcsvm(x_train, double(y_train(:,2)==k-1),'KernelFonction','RBF'));
end

%% Get the posterior probability matrix for the predictions

%# get probability estimates of test instances using each model
numTest = size(x_val,1);
prob = zeros(numTest,numLabels);
for k=1:numLabels
    fprintf('Computing SVM for class %i\n',k);
    [~,p] = predict(model{k}, x_val);
    prob(:,k) = p(:,model{k}.ClassNames==1);    %# probability of class==k
end

%# predict the class with the highest probability
[~,pred] = max(prob,[],2);
acc = sum(pred == y_val) ./ numel(y_val);    %# accuracy
C = confusionmat(y_val, pred);                   %# confusion matrix
fprintf('Resulting accuracy : %f\n',acc);

