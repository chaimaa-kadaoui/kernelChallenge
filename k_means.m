function [labels, C] = k_means(X, k, maxiter)
    n = size(X,1);
    labels = randi(k,1,n);
    last = 0;
    iter = 0;
    while any(labels ~= last) && iter<maxiter
        E = sparse(1:n,labels,1,n,k,n);  % transform label into indicator matrix
        C = (E*spdiags(1./sum(E,1)',0,k,k))'*X;    % compute centers 
        last = labels;
        [val,labels] = max(bsxfun(@minus,C*X',dot(C,C,2)/2),[],1); % assign labels
        distorsion = dot(X(:),X(:))-2*sum(val);
        fprintf('Distorsion at iteration %i: %f\n',iter, distorsion);
        iter = iter+1;
    end
    labels = labels';
end