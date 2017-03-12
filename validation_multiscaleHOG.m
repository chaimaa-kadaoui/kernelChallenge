addpath(genpath(pwd))

%% Load data
x_train_total = csvread('Xtr.csv');
x_train_total = x_train_total(:,1:end-1);
y_train_total = csvread('Ytr.csv');

%Moving input data to grayscale
x_train_total = reshape(x_train_total, [size(x_train_total,1),1024,3]);
x_train_total = mean(x_train_total,3);

%% Cross validation of Multi Scale HOG features
kernels = {'RBF','intersection','hellinger','chi2'};
p_norms = [0,1,2];

accs = zeros(length(kernels),length(p_norms),10);
mean_accs = zeros(length(kernels),length(p_norms));

for i=1:length(kernels)
    kernel = char(kernels{i});
    for j=1:length(kernels)
        p_norm = p_norms(j);
        fprintf('Validation of kernel %s with normalization %i',kernel,p_norm);
        [accs(i,j,:),mean_accs(i,j)] = validate_multiscaleHOG(x_train_total, y_train_total, kernel, p_norm);
    end
end

%% Extract best parameters
[best_acc,p] = max(mean_accs(:));
[best_i, best_j] = ind2sub(size(mean_accs),p);
fprintf('Best accuracy (%f) obtained with kernel %s and normalization %i',best_acc, char(kernels{best_i}), p_norms(best_j));