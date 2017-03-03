function [alpha_y, bias] = fitcsvm_kernel(gram_matrix, y_bin, C)
    % Fit a SVM to the Kernel matrix passed as argument
    
    % gram_matrix is a NxN gram matrix already computed for the train data
    % with the desired kernel
    % y_bin is the binary labels of size Nx1 with +1 or -1 values
    % C is the BoxConstraint
    
    % Dimension
    N = size(gram_matrix, 1); 
    
    % Matrices
    H = diag(y_bin)*gram_matrix*diag(y_bin);
    f = -ones(N,1);
    A = [];
    b = [];
    Aeq = y_bin';
    beq = 0;
    lb = zeros(N,1);
    ub = C*ones(N,1);
    
    %Quadratic program
    options = optimoptions('quadprog','Display','off');
    alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);
    alpha_y = alpha.*y_bin;
    
    %Bias
    mask = (0<alpha) & (alpha<C);
    sub_y = y_bin(mask);
    sub_gram = gram_matrix(:,mask);
    bias = mean(sub_y' - alpha_y'*sub_gram);
end