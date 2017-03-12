function [alpha, lambda] = KPCA(K,T)
    % Build centered kernel matrix
    n = length(K);
    S = sum(K)/n;
    J = ones(n,1) * S;
    K = K-(J+J')+sum(S)/n;
    
    if ~issymmetric(K)
        disp('Non symmetric matrix');
    end
    
    % Compute eigenvectors associeted to T highest eigenvalues
    [alpha, lambda] = eigs(K,T);
    lambda = diag(abs(lambda));
    alpha = bsxfun(@rdivide, alpha, abs(sqrt(lambda))');
    if ~isreal(alpha) || ~isreal(lambda)
        disp('Eigenvalues or eigenvectors not real!');
        disp('Eigenvalues');
        disp(lambda);
    end
end
