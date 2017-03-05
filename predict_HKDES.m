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
[x_patches, x_statistics] = processKDES(x_train_total,5,3,6,2,0.8,0.2,8,2,200,50,200);
save('x_patches.mat','x_patches');
save('x_statistics.mat','x_statistics');

[x_patches_test, x_statistics_test] = processKDES(x_test,5,3,6,2,0.8,0.2,8,2,200,50,200);
save('x_patches_test.mat','x_patches_test');
save('x_statistics_test.mat','x_statistics_test');

[X_G,X_C,X_S] = create_basis(x_patches,8,2,200,50,200,1000);
save('X_basis.mat', 'X_G','X_C','X_S');

x_train_total_p = processHKDES(x_patches,x_statistics,X_G,X_C,X_S,1,1,1,1,0.5,8,2,2000,500,2000);
save('x_HKDES.mat', x_train_total_p);

x_test_p = processHKDES(x_patches_test,x_patch_statistics_test,X_G,X_C,X_S,1,1,1,1,0.5,8,2,2000,500,2000);
save('x_HKDES_test.mat', x_test_p);

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


