%Test harris

%Load some data
x_train_total = csvread('Xtr.csv');
x_train_total = x_train_total(:,1:end-1);

%% Load an image
test_img = mean(reshape(x_train_total(6,:),[32 32 3]),3);
%test_img = open('girl.png');
%test_img = mean(test_img.girl,3);

figure(1);
subplot(1,3,1);
imshow(test_img,[]);

%Test harris
img_harris = harris_corner_detector(test_img,1,2,0.0001,9);
subplot(1,3,2);
imshow(img_harris,[]);

%Test anms
max_val = max(test_img(:));
min_val = min(test_img(:));
correc_img = uint8(255*(test_img-min_val)/(max_val - min_val));
new_img = cat(3, correc_img, correc_img, correc_img);

points_of_interest = anms(img_harris,10,0.99);
for k=1:size(points_of_interest,1)
    ind = points_of_interest(k,:);
    new_img(ind(1),ind(2),:) = [255 0 0];
end
subplot(1,3,3);
imshow(new_img,[]);

%% Test HOG Computation

% Load an image
test_img = mean(reshape(x_train_total(6,:),[32 32 3]),3);
% Harris corner detector
img_harris = harris_corner_detector(test_img,1,2,0.0001,9);
% ANMS
points_of_interest = anms(img_harris,10,0.99);
%HOG with centers
[x_p] = processHOGrad(reshape(test_img,[1,1024]), 7, points_of_interest, 2);
