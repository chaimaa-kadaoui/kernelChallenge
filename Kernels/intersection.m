function G = intersection(U,V)
    % intersection_kernel = @(x,Y) sum(bsxfun(@min,x,Y),2);
    % G = pdist2(U,V,intersection_kernel);
    
    G = sum(bsxfun(@min,permute(U,[1 3 2]),permute(V,[3 1 2])),3);
end