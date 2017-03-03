function [alpha, lambda] = KPCA(K,T)
    % Build centered kernel matrix
    n = length(K);
    S = sum(K)/n;
    J = ones(n,1) * S;
    K = K-J-J'+sum(S)/n;
    
    % Compute eigenvectors associeted to T highest eigenvalues
    [alpha, lambda] = eigs(K,T);
    lambda = diag(abs(lambda));
    alpha = bsxfun(@rdivide, alpha,sqrt(lambda)');
end
