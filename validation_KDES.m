addpath(genpath(pwd))

%% Load data
x_train_total = csvread('Xtr.csv');
x_train_total = x_train_total(:,1:end-1);
y_train_total = csvread('Ytr.csv');

%Moving input data to grayscale
x_train_total = reshape(x_train_total, [size(x_train_total,1),1024,3]);
x_train_total = mean(x_train_total,3);

%% Cross validation of KDES features
params = [[8,8,50,50,50];[8,8,100,100,100];[8,8,200,200,200];
          [8,2,50,50,50];[8,2,100,100,100];[8,2,200,200,200];];

accs = zeros(length(params),10);
mean_accs = zeros(length(params));

for l=1:length(params)
    [accs(l,:),mean_accs(l)] = validateKDES(x_train_total, y_train_total, params(l,:));
end

%% Extract best parameters
[best_acc,p] = max(mean_accs(:));
fprintf('Best accuracy (%f) obtained with stride %i and T %i',best_acc, params(p,2),params(p,3));
