function [x_p] = processKDES(x,gamma_o,gamma_p,gamma_c,gamma_b,eps_g,eps_s,window_size,stride,T)
    x = reshape(x,[size(x,1),32,32]);
    l = floor((32-window_size)/stride)+1;
    
    x_p = zeros(size(x,1)*l^2,3*T);
    % Creation of the basis points of k_o
    grid_o = linspace(0,2*pi,25)';
    X_o = [cos(grid_o) sin(grid_o)];
    % Creation of the basis points of k_c
    X_c = linspace(0,1,5)';
    % Creation of the basis points of k_p
    grid_p = linspace(0,1,5);
    [meshX,meshY] = meshgrid(grid_p);
    X_p = [meshX(:) meshY(:)];
    % Creation of the basis points of k_b
    grid_b = dec2bin((0:2^8-1),8);
    X_b = zeros(2^8,8);
    for s=1:8
        X_b(:,s) = str2num(grid_b(:,s)); %#ok<ST2NM>
    end
    
    % Gram matrix of k_o
    K_o = rbf(X_o,X_o,gamma_o);
    % Gram matrix of k_c
    K_c = rbf(X_c,X_c,gamma_c);
    % Gram matrix of k_p
    K_p = rbf(X_p,X_p,gamma_p);
    % Gram matrix of k_b
    K_b = rbf(X_b,X_b,gamma_b);
    
    % KPCA on the basis vectors phi_o x phi_p
    alpha_op = KPCA(K_o,K_p,T);
    % KPCA on the basis vectors phi_c x phi_p
    alpha_cp = KPCA(K_c,K_p,T);
    % KPCA on the basis vectors phi_b x phi_p
    alpha_bp = KPCA(K_b,K_p,T);
    
    % Position vectors z
    grid_z = linspace(0,1,window_size);
    [meshX,meshY] = meshgrid(grid_z);
    Z = [meshY(:) meshX(:)];
    
    % Loop over the images
    for i=1:size(x,1)
        % Extract the image
        image = squeeze(x(i,:,:));
        % Compute the directions of the gradient
        [Gmag,Gdir] = imgradient(image);
        % Loop over the patches in the image
        for row=1:l
            for col=1:l
                window_r = (row-1)*stride+(1:row*window_size);
                window_c = (col-1)*stride+(1:col*window_size);
                % Computation of m_tilde (formula (2))
                m = reshape(Gmag(window_r,window_c),window_size^2,1);
                m = m/sqrt(sum(m.^2)+eps_g);
                % Computation of theta_tilde (formula (6))
                theta = reshape(Gdir(window_r,window_c),window_size^2,1);
                theta = [cosd(theta) sind(theta)];
                % Computation of c (formula (7))
                c = reshape(image(window_r,window_c),window_size^2,1);
                % Computation of s_tilde and b (formula(8))
                s = zeros(window_size^2,1);
                b = zeros(window_size^2,8);
                for z_i = 1:window_size
                    for z_j = 1:window_size
                        std_window_r = (row-1)*stride+(z_i-1:z_i+1);
                        std_window_r = std_window_r(std_window_r>0 && std_window_r<=32);
                        std_window_c = (col-1)*stride+(z_j-1:z_j+1);
                        std_window_c = std_window_c(std_window_c>0 && std_window_c<=32);
                        s_pixels = image(std_window_r,std_window_c);
                        s_pixels(find(s_pixels==image((row-1)*stride+z_i,(col-1)*stride+z_j),1)) = [];
                        s((z_i-1)*window_size+z_j) = std(s_pixels);
                        b_pixels = zeros(3,3);
                        b_pixels(std_window_r-(row-1)*stride-z_i+2,std_window_c-(col-1)*stride-z_j+2) = image(std_window_r,std_window_c)>image((row-1)*stride+z_i,(col-1)*stride+z_j);
                        b_pixels = reshape(b_pixels,9,1);
                        b_pixels(5) = [];
                        b((z_i-1)*window_size+z_j) = b_pixels;
                    end
                end
                s = s/sqrt(sum(s.^2)+eps_s);
                
                % Gram matrix of k_o
                K_o = rbf(theta,X_o,gamma_o);
                % Gram matrix of k_c
                K_c = rbf(c,X_c,gamma_c);
                % Gram matrix of k_p
                K_p = rbf(Z,X_p,gamma_p);
                % Gram matrix of k_b
                K_b = rbf(b,X_b,gamma_b);
                
                % Compute F_grad(P) (formula (12))
                F_grad = kernel_descriptors(m, K_o, K_p);
                % Compute F_col(P) (derived from formula (7))
                F_col = kernel_descriptors(ones(window_size^2,1), K_c, K_p);
                % Compute F_shape(P) (derived from formula (8))
                F_shape = kernel_descriptors(s, K_b, K_p);
                
                % Put together patch features
                x_p((row-1)*l+col,:) = [F_grad F_col F_shape];
            end
        end
    end
end