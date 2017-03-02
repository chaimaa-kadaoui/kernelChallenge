function [F] = kernel_descriptors(c,K1,K2,alpha)
    d1 = size(K1,2);
    d2 = size(K2,2);
    T = size(alpha,2);
    F = zeros(1,T);
    alpha = reshape(alpha,d1,d2,T);
    
    parfor t=1:T
        % Features computation (formula (12))
        F(t) = sum(alpha(:,:,t)*K2'.*K1',1)*c;
    end
end