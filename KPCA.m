function [alpha] = KPCA(K1,K2,T)
    Kc = kron(K1,K2);
    Kc = bsxfun(@minus, Kc, 2*sum(Kc,1))+sum(Kc(:));
    [~,alpha] = eigs(Kc,T);
end
