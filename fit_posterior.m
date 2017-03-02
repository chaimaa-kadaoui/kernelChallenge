function [post_proba] = fit_posterior(score, A, B)    
    % Posterior Probabilities
    % The first column is for the class 1: P(y=1|score)
    % The second is for the class -1: P(y=-1|score) = 1 - P(y=1|score)
    N = size(score, 1);
    post_proba = zeros(N,2);
    post_proba(:,1) = 1./(1+exp(A*score+B));
    post_proba(:,2) = 1-post_proba(:,1);
end