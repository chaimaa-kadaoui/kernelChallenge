function [y_new, score] = predict_svm(gram_matrix, alpha_y, bias)
    score = alpha_y'*gram_matrix + bias;
    score = score';
    y_new = sign(score);
end