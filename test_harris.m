%Test harris

%Load some data
x_train_total = csvread('Xtr.csv');
x_train_total = x_train_total(:,1:end-1);

%Load an image
test_img = mean(reshape(x_train_total(1,:),[32 32 3]),3);
figure(1);
subplot(1,2,1);
imshow(test_img,[]);

%Test harris
img_harris = harris_corner_detector(test_img,1,2,0.1);
subplot(1,2,2);
imshow(img_harris,[]);