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
stride = 4;

x_train_total_p = processHOG(x_train_total,window_size,stride);
x_test_p = processHOG(x_test,window_size,stride);

%% Train SVM

% train one-against-all models
numLabels = length(unique(y_train_total(:,2)));
model = cell(numLabels,1);
for k=1:numLabels
    fprintf('Computing SVM for class %i\n',k);
    model{k} = fitSVMPosterior(fitcsvm(x_train_total_p, double(y_train_total(:,2)==k-1),'KernelFunction','RBF'));
end

%% Get the posterior probability matrix for the predictions

% get probability estimates of test instances using each model
numTest = size(x_test,1);
prob = zeros(numTest,numLabels);
for k=1:numLabels
    fprintf('Computing posteriors for class %i\n',k);
    [~,p] = predict(model{k}, x_test_p);
    prob(:,k) = p(:,model{k}.ClassNames==1);    % probability of class==k
end

%% Do the prediction

% predict the class with the highest probability
[~,pred] = max(prob,[],2);
pred = pred-1;
pred = [(1:numTest)' pred];

% write prediction to file
path = './results/Yte.csv';
csvfile = fopen(path,'w');
fprintf(csvfile,'Id,Prediction\n');
fclose(csvfile);
dlmwrite (path, pred, '-append');


