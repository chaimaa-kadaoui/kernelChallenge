function [labels, C] = k_means(X, k)
    rng(1234);
    
    n = size(X,1);
    labels = randi(k,1,n);
    last = 0;
    iter = 0;
    while any(labels ~= last)
        [u,~,labels] = unique(labels);
        labels = labels';
        k = numel(u);
        E = sparse(1:n,labels,1,n,k,n);
        C = (E*spdiags(1./sum(E,1)',0,k,k))'*X;
        last = labels;
        [val,labels] = max(bsxfun(@minus,C*X',dot(C,C,2)/2),[],1);
        distorsion = dot(X(:),X(:))-2*sum(val);
        fprintf('Distorsion at iteration %i: %f\n',iter, distorsion);
        iter = iter+1;
    end
    labels = labels';
end