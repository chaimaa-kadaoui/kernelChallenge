%% Load Data

x_test = csvread('Xte.csv');
x_test=x_test(:,1:end-1);
x_train = csvread('Xtr.csv');
x_train=x_train(:,1:end-1);
y_train = csvread('Ytr.csv');

%Moving input data to grayscale
x_train = reshape(x_train, [5000,1024,3]);
x_train = mean(x_train,3);

%Moving input data to grayscale
x_test = reshape(x_test, [2000,1024,3]);
x_test = mean(x_test,3);

%% Train SVM

%# train one-against-all models
numLabels = length(unique(y_train(:,2))); %Handle class 0
model = cell(numLabels,1);
for k=1:numLabels
    model{k} = fitSVMPosterior(fitcsvm(x_train, double(y_train(:,2)==k)));
end

%% Get the posterior probability matrix for the predictions

%# get probability estimates of test instances using each model
numTest = size(x_test,1);
prob = zeros(numTest,numLabels);
for k=1:numLabels
    [~,p] = predict(model{k}, x_test);
    prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
end

%# predict the class with the highest probability
[~,pred] = max(prob,[],2);
acc = sum(pred == testLabel) ./ numel(testLabel)    %# accuracy
C = confusionmat(testLabel, pred)                   %# confusion matrix

