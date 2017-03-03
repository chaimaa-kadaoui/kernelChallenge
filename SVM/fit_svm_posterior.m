function [A, B] = fit_svm_posterior(gram_matrix, y_bin, C)
    % Fit the posterior probability of the SVM
    % First compute a new score using cross validation
    % Second learn A & B using a model-trust minimization
    
    % gram_matrix is a NxN matrix Gram matrix
    % y_bin is a Nx1 vecotr of labels +1 and -1
    % C is the BoxConstraint
    
    score_cv = get_score_crossval(gram_matrix, y_bin, C);
    [A, B] = fit_sigmoid(score_cv, y_bin);
end