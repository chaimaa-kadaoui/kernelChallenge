%% Load x_train_total_p from file x_HKDES.mat
load('x_hkdes.mat')

%% Load data
y_train_total = csvread('Ytr.csv');

%% Cross validation of HKDES features
params = [[800,50,800]; [800, 80, 800]; [700,30,700]; [900, 20, 900]];

accs = zeros(length(params),10);
mean_accs = zeros(length(params),1);

parfor l=1:length(params)
	x_train_cut = x_train_total_p(:,[1:params(l,1), 2000+(1:params(l,2)), 2500+(1:params(l,3))]);
    [accs(l,:),mean_accs(l)] = validateHKDES(x_train_cut, y_train_total);
end

%% Extract best parameters
[best_acc,p] = max(mean_accs(:));
fprintf('Best accuracy (%f) obtained with set of parameters %i',best_acc, p);
disp(params(p,:));
