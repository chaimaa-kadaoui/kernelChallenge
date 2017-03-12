function G = hellinger(U,V)
    U = sqrt(U);
    V = sqrt(V);
    G = U*V';
end