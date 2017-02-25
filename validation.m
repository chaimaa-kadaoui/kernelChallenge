%% Load data
x_train_total = csvread('Xtr.csv');
x_train_total = x_train_total(:,1:end-1);
y_train_total = csvread('Ytr.csv');

%Moving input data to grayscale
x_train_total = reshape(x_train_total, [size(x_train_total,1),1024,3]);
x_train_total = mean(x_train_total,3);

%% Cross validation of HOG features
window_size = 8;
strides = 8;
kernels = {'intersection','hellinger','chi2'};
p_norms = [1,1,0];

accs = zeros(length(strides),length(kernels),10);
mean_accs = zeros(length(strides),length(kernels));

for i=1:length(strides)
    stride = strides(i);
    for j=1:length(kernels)
        kernel = char(kernels{j});
        p_norm = p_norms(j);
        [accs(i,j,:),mean_accs(i,j)] = validateHOG(x_train_total, y_train_total, 8, stride, kernel, p_norm);
    end
end

%% Extract best parameters
[best_acc,p] = max(mean_accs(:));
[best_i, best_j] = ind2sub(size(mean_accs),p);
fprintf('Best accuracy (%f) obtained with kernel %s, window_size 8 and stride %i',best_acc, char(kernels{best_j}), strides(best_i));