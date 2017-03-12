function G = chi2(U,V)
    G = zeros(size(U,1),size(V,1));
    for i=1:size(V,1)
        d = bsxfun(@minus, U, V(i,:));
        s = bsxfun(@plus, U, V(i,:));
        G(:,i) = sum(d.^2 ./ (s/2+eps), 2);
    end
    G = 1 - G;
end