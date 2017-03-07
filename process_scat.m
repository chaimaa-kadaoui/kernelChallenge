function [ x_p ] = process_scat( x, y )
%Process a scattering network feature extraction on x
%   

addpath 'scatnet-0.2';
addpath_scatnet();
addpath 'OLS';

x_r = reshape(x,size(x,1),32,32);
filt_opt.J = 4;
filt_opt.L = 8;
scat_opt.oversampling = 2;
scat_opt.M = 2;
Wop = wavelet_factory_2d(size(squeeze(x_r(1,:,:))),filt_opt,scat_opt);
S_mat = sum(sum(format_scat(scat(squeeze(x_r(1,:,:)),Wop)),2),3);
scatter = sum(format_scat(scat(squeeze(x_r(1,:,:)),Wop)),2);

x_p = zeros(size(x,1),numel(scatter));
parfor i = 1:size(x,1)
    scatter_i = sum(format_scat(scat(squeeze(x_r(i,:,:)),Wop)),2);
    x_p(i,:) = reshape(scatter_i,[1,numel(scatter_i)]);
    if mod(i,1000) == 0
        fprintf('Progress : %f%%\n',i/size(x,1));
    end
end

threshold = 0.1;
[x_ols, ind] = ols(x_p,y(:,2),floor(numel(scatter)/30));

x_p = x_p(:,ind);

end

