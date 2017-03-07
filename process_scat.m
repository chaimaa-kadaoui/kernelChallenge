function [ x_p ] = process_scat( x )
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

x_p = zeros(size(x,1),length(S_mat));
parfor i = 1:size(x,1)
    x_p(i,:) = sum(sum(format_scat(scat(squeeze(x_r(i,:,:)),Wop)),2),3)';
    if mod(i,1000) == 0
        fprintf('Progress : %f%%\n',i/size(x,1));
    end
end

end

