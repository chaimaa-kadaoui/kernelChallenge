function G = rbf(U,V, gamma)
    G = pdist2(U,V,'squaredeuclidean');
    G = exp(-gamma*G);
end