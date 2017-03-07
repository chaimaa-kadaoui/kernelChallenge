function G = rbf_old(U,V, gamma)
    G = pdist2(U,V,'euclidean');
    G = exp(-gamma*G);
end