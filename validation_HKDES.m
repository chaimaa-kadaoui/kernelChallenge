%% Load data
x_train_total = csvread('Xtr.csv');
x_train_total = x_train_total(:,1:end-1);
y_train_total = csvread('Ytr.csv');

%Moving input data to grayscale
x_train_total = reshape(x_train_total, [size(x_train_total,1),1024,3]);
x_train_total = mean(x_train_total,3);

%% Extraction of patch kernel descriptors
[x_patches, x_statistics] = processKDES(x_train_total,5,3,6,2,0.8,0.2,8,2,200,50,200);

%% Cross validation of HKDES features
params = [[2,1,1,1,0.2];[2,4,1,1,0.3]]

accs = zeros(length(params),10);
mean_accs = zeros(length(params));

for l=1:length(params)
    [accs(l,:),mean_accs(l)] = validateKDES(x_patches, x_statistics, y_train_total, params(l,:));
end

%% Extract best parameters
[best_acc,p] = max(mean_accs(:));
fprintf('Best accuracy (%f) obtained with gamma_o %i gamma_p %i',best_acc, params(p,1), params(p,2));
