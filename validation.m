%% Load data
x_train_total = csvread('Xtr.csv');
x_train_total = x_train_total(:,1:end-1);
y_train_total = csvread('Ytr.csv');

%Moving input data to grayscale
x_train_total = reshape(x_train_total, [size(x_train_total,1),1024,3]);
x_train_total = mean(x_train_total,3);

%% Cross validation of HOG features
window_sizes = [8,12,16];
strides = [2,4];

accs = zeros(length(window_sizes),length(strides),10);
mean_accs = zeros(length(window_sizes),length(strides));

for i=1:length(window_sizes)
    window_size = window_sizes(i);
    for j=1:length(strides)
        stride = strides(j);
        [accs(i,j,:),mean_accs(i,j)] = validateHOG(x_train_total, y_train_total, window_size,stride);
    end
end

%% Extract best parameters
[best_acc,p] = max(mean_accs(:));
[best_i, best_j] = ind2sub(size(mean_accs),p);
fprintf('Best accuracy (%f) obtained with window_size %i and stride %i',best_acc, window_sizes(best_i), strides(best_j));