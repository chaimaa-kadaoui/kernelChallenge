function [post_proba] = fit_posterior(gram_matrix, alpha_y, bias)
    [~,score] = predict_diy(gram_matrix, alpha_y, bias);
    
    % Probabilities
    % The first column is for the class 1
    % The second is for the class -1
    N = size(gram_matrix, 2);
    post_proba = zeros(N,2);
    post_proba(:,1) = 1./(1+exp(-score));
    post_proba(:,2) = 1-post_proba(:,1);
end