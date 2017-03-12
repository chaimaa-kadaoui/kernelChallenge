addpath(genpath(pwd))

%% Load x_train_total_p from file x_HKDES.mat
load('x_HKDES_all.mat')

%% Load data
y_train_total = csvread('Ytr.csv');

x_center = bsxfun(@minus, x_all, mean(x_all,1));
x_train_total_p = x_center(1:5000,:);

%% Cross validation of HKDES features
params = [[750,20,750,1]; [775,22,775,1]];

accs = zeros(size(params,1),10);
mean_accs = zeros(size(params,1),1);

parfor l=1:size(params,1)
	x_train_cut = x_train_total_p(:,[1:params(l,1), 1000+(1:params(l,2)), 1200+(1:params(l,3))]);
    [accs(l,:),mean_accs(l)] = validateHKDES(x_train_cut, y_train_total, params(l,:));
end

%% Extract best parameters
[best_acc,p] = max(mean_accs(:));
fprintf('Best accuracy (%f) obtained with set of parameters %i',best_acc, p);
disp(params(p,:));
