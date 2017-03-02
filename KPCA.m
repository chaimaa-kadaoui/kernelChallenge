function [alpha] = KPCA(K1,K2,T)
    % Build centered kernel matrix
    % (formula from the one of the course, different in the paper)
    Kc = kron(K2,K1);
    n = length(Kc);
    S = sum(Kc)/n;
    J = ones(n,1) * S;
    Kc = Kc-J-J'+sum(S)/n;
    
    % Compute eigenvectors associeted to T highest eigenvalues
    [alpha, lambda] = eigs(Kc,T);
    alpha = bsxfun(@rdivide, alpha,sqrt(diag(lambda))');
end
