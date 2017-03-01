function [ harris_img ] = harris_corner_detector( img,sigma_derivative,sigma_integration, epsilon, sz )
%Computes the harris corner detector
%   sigma_derivation : std deviation for computing the derivation operator
%   
raw_deriv = [-0.5 0 0.5];
x = linspace(-sz / 2, sz / 2, sz);
gaussFilter = exp(-x .^ 2 / (2 * sigma_derivative ^ 2));
gaussFilter = gaussFilter / sum (gaussFilter);
smooth_deriv = conv(raw_deriv,gaussFilter);

%Computing the image derivative
Ix = conv2(img,smooth_deriv,'same');
Iy = conv2(img,smooth_deriv','same');

%Extra smoothing filter
x_i = linspace(-sz / 2, sz / 2, sz);
gaussFilter_i = exp(-x .^ 2 / (2 * sigma_integration ^ 2));
gaussFilter_i = gaussFilter / sum (gaussFilter);

%Compute product images with extra smoothing
Ix_2 = conv2(Ix.*Ix,gaussFilter_i,'same');
Iy_2 = conv2(Iy.*Iy,gaussFilter_i,'same');
Ixy = conv2(Ix.*Iy,gaussFilter_i,'same');

%Computing the harris function
harris_img = (Ix_2.*Iy_2 - Ixy.^2)./(Ix_2 + Iy_2 + epsilon);


end

