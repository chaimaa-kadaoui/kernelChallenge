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

window_size = 8;
stride = 2;

x_train_total_p = processHOG(x_train_total,window_size,stride,2);
x_test_p = processHOG(x_test,window_size,stride,2);

%% Kernel

gamma = 1;
gram_train = rbf(x_train_total_p,x_train_total_p,gamma);
gram_test = rbf(x_train_total_p,x_test_p,gamma);
C = 1;


%% Train SVM

% train one-against-all models
numLabels = length(unique(y_train_total(:,2)));
model_diy = cell(numLabels,1);
model = cell(numLabels,1);
parfor k=1:numLabels
    fprintf('Computing SVM for class %i\n',k);
    % DIY
    fprintf('DIY \n');
    y_bin = zeros(size(y_train_total,1),1);
    y_bin(y_train_total(:,2)==k-1)=1;
    y_bin(y_train_total(:,2)~=k-1)=-1;
    [alpha_y, bias] = fitcsvm_kernel(gram_train, y_bin, C);
    [A, B] = fit_svm_posterior(gram_train, y_bin, C, 10);
    model_diy{k}.alpha_y = alpha_y;
    model_diy{k}.bias = bias;
    model_diy{k}.A = A;
    model_diy{k}.B = B;
end

%% Get the posterior probability matrix for the predictions

% get probability estimates of test instances using each model
numTest = size(x_test,1);
prob_diy = zeros(numTest,numLabels);
prob = zeros(numTest,numLabels);
parfor k=1:numLabels
    fprintf('Computing posteriors for class %i\n',k);
    fprintf('DIY \n');
    [~, score] = predict_svm(gram_test, model_diy{k}.alpha_y, model_diy{k}.bias);
    post_proba = get_posterior(score, model_diy{k}.A, model_diy{k}.B);
    prob_diy(:,k) = post_proba(:,1);    %# probability of class==k
end

%% Do the prediction

% predict the class with the highest probability
[~,pred_diy] = max(prob_diy,[],2);
pred_diy = pred_diy-1;
pred_diy = [(1:numTest)' pred_diy];

% write prediction to file
path = './results/Yte_DIY_10CV.csv';
csvfile = fopen(path,'w');
fprintf(csvfile,'Id,Prediction\n');
fclose(csvfile);
dlmwrite (path, pred_diy, '-append');

