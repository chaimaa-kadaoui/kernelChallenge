function [x_p] = processHKDES(x_patches,x_patch_statistics, X_G, X_C, X_S,gamma_C,gamma_Fg,gamma_Fc, gamma_Fs, eps_h, window_size, stride, T_G, T_C, T_S)
    l = floor((32-window_size)/stride)+1;
    
    x_p = zeros(size(x_patches,1),T_G+T_C+T_S);
    
    T_grad = size(X_G,2);
    T_col = size(X_C,2);
    T_shape = size(X_S,2);
    T = T_grad+T_col+T_shape;
    
    disp('Creating basis points');
    % Creation of the basis points of k_C
    grid_c = linspace(0,1,5);
    [meshX,meshY] = meshgrid(grid_c);
    X_c = [meshX(:) meshY(:)];
    
    disp('Building Gram matrices of basis vectors');
    % Gram matrix of k_C
    K_c = rbf(X_c,X_c,gamma_C);
    % Gram matrix of k_Fg
    K_G = rbf(X_G,X_G,gamma_Fg);
    % Gram matrix of k_Fc
    K_C = rbf(X_C,X_C,gamma_Fc);
    % Gram matrix of k_Fs
    K_S = rbf(X_S,X_S,gamma_Fs);
    
    disp('Performing KPCA on basis vectors');
    % KPCA on the basis vectors phi_C
    [alpha_c,lambda_c] = KPCA(K_c,20);
    % KPCA on the basis vectors phi_Fg
    [alpha_G,lambda_G] = KPCA(K_G,200);
    % KPCA on the basis vectors phi_Fc
    [alpha_C,lambda_C] = KPCA(K_C,200);    
    % KPCA on the basis vectors phi_Fs
    [alpha_S,lambda_S] = KPCA(K_S,200);  
    
    disp('Computing eigenvectors of Gram matrices');
    % Computing eigenvectors of k_C x k_Fg
    alpha_grad = kron(alpha_c,alpha_G);
    lambda_grad = kron(lambda_c,lambda_G);
    [~,order] = sort(lambda_grad,'descend');
    alpha_grad = alpha_grad(:,order(1:T_G));
    
    % Computing eigenvectors of k_C x k_Fc
    alpha_col = kron(alpha_c,alpha_C);
    lambda_col = kron(lambda_c,lambda_C);
    [~,order] = sort(lambda_col,'descend');
    alpha_col = alpha_col(:,order(1:T_C));
    
    % Computing eigenvectors of k_C x k_Fs
    alpha_shape = kron(alpha_c,alpha_S);
    lambda_shape = kron(lambda_c,lambda_S);
    [~,order] = sort(lambda_shape,'descend');
    alpha_shape = alpha_shape(:,order(1:T_S));
    
    % Position vectors C_A
    grid_z = linspace(0,1,l);
    [meshX,meshY] = meshgrid(grid_z);
    Z = [meshX(:) meshY(:)];
    
    W_total = x_patch_statistics(:,1:2:end);
    S_total = x_patch_statistics(:,2:2:end);
    
    % Loop over the images
    parfor i=1:size(x_patches,1)
        fprintf('Computing hierarchical kernel descriptors for image %i\n',i);
        
        W = W_total(i,:)';
        W = W/sqrt(sum(W.^2)+eps_h);

        S = S_total(i,:)';
        S = S/sqrt(sum(S.^2)+eps_h);
        
        F = reshape(x_patches(i,:),T,l^2)'
        Fg = F(:,1:T_grad);
        Fc = F(:,T_grad+(1:T_col));
        Fs = F(:,T_grad+T_col+(1:T_shape));
        
        % Gram matrix of k_b
        K_CC = rbf(Z,X_c,gamma_C);
        % Gram matrix of k_Fg
        K_Fg = rbf(Fg,X_G,gamma_Fg);
        % Gram matrix of k_Fc
        K_Fc = rbf(Fc,X_C,gamma_Fc);
        % Gram matrix of k_Fs
        K_Fs = rbf(Fs,X_S,gamma_Fs);
          
        % Compute Fg(I) (formula (10))
        F_Hgrad = kernel_descriptors(W, K_Fg, K_CC, alpha_grad);
        % Compute F_col(P)
        F_Hcol = kernel_descriptors(ones(l^2,1), K_Fc, K_CC, alpha_col);
        % Compute F_shape(P)
        F_Hshape = kernel_descriptors(S, K_Fs, K_CC, alpha_shape);
                
        % Put together image features
        x_p(i,:) = [F_Hgrad F_Hcol F_Hshape];
    end
end