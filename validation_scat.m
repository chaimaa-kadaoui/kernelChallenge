addpath(genpath(pwd))

%% Load data
x_train_total = csvread('Xtr.csv');
x_train_total = x_train_total(:,1:end-1);
y_train_total = csvread('Ytr.csv');

%Moving input data to grayscale
x_train_total = reshape(x_train_total, [size(x_train_total,1),1024,3]);
x_train_total = mean(x_train_total,3);

%% Cross validation of HOG features
%params = [[6,2];[6,4];[8,2];[8,4];[8,6];[12,4];[12,5]];
params = [[8,2]];
accs = zeros(length(params),10);
mean_accs = zeros(length(params));

for l=1:length(params)
    window_size = params(l,1);
    stride = params(l,2);
    [accs(l,:),mean_accs(l)] = validate_scat(x_train_total, y_train_total,'RBF');
end

%% Extract best parameters
[best_acc,p] = max(mean_accs(:));
fprintf('Best accuracy (%f) obtained with window_size %i and stride %j',best_acc, params(p,1), params(p,2));
