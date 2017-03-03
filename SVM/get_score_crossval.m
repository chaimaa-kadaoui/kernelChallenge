function [score_train] = get_score_crossval(gram_matrix, y_bin, C)
    % Using a 3-fold cross-validation
    % Compute a new set of score for training to avoid overfitting
    
    % Initialization
    N = size(gram_matrix,1);
    score_train = zeros(N,1);
    
    % Split
    splitPart = 3;
    splitSize = floor(N/splitPart);
    
    % Cross-validation
    for valPart=1:splitPart
        part_train = [1:(valPart-1)*splitSize, valPart*splitSize+1:N];
        part_val = (valPart-1)*splitSize+1:valPart*splitSize;
        gram_train = gram_matrix(part_train,part_train);
        y_train = y_bin(part_train);
        gram_val = gram_matrix(part_train,part_val);
        [alpha_y, bias] = fitcsvm_kernel(gram_train, y_train, C);
        [~, score] = predict_svm(gram_val, alpha_y, bias);
        score_train(part_val) = score;
    end
end