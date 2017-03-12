function [X_G,X_C,X_S] = create_basis(x_patches,window_size, stride, T_grad,T_col,T_shape,n_clusters)
    l = floor((32-window_size)/stride)+1;

    T = T_grad+T_col+T_shape;
    patch_features = zeros(size(x_patches,1)*l^2,T);
    for i=1:size(x_patches,1)
        patch_features((i-1)*l^2+1:i*l^2,:) = reshape(x_patches(i,:),T,l^2)';
    end
    
    % Creation of the basis points of k_Fg (using kmeans)
    disp('Creating basis points for gradient patch kernel (kmeans)');
    [~, X_G] = k_means(patch_features(1:8:end,1:T_grad), n_clusters);
    % Creation of the basis points of k_Fc (using kmeans)
    disp('Creating basis points for intensity patch kernel (kmeans)');
    [~, X_C] = k_means(patch_features(1:8:end,T_grad+(1:T_col)), n_clusters);
    % Creation of the basis points of k_Fs (using kmeans)
    disp('Creating basis points for shape patch kernel (kmeans)');
    [~, X_S] = k_means(patch_features(1:8:end,T_grad+T_col+(1:T_shape)), n_clusters);
end