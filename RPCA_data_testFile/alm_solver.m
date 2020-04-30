function [X, L, S, Y, Z, res, iter] = alm_solver(X)

    [r, c] = size(X);
    unobserved = isnan(X);
    lambda = 1 / sqrt(max(r, c));
    mu = 10 * lambda;
    tol = 1e-4;
    L = zeros(r, c);
    LNot = zeros(r, c);
    S = zeros(r, c);
    SNot = zeros(r, c);
    Y = zeros(r, c);
    t = 1;
    tNot = 1;
    
    iter = 0;
    res = 1;
    while res > tol
        iter = iter + 1;
        
        YL = L + ((tNot - 1) / t) * (L - LNot);
        YS = S + ((tNot - 1) / t) * (S - SNot);
        GL = YL - 0.5 * (YL + YS - X);
        LNot = L;
        SNot = S;
        L = Do(mu / 2, GL);
        GS = YS - 0.5 * (YL + YS - X);
        S = So((lambda * mu) / 2, GS);
        tNot = t;
        t = (1 + sqrt((4 * (t^2)) + 1)) / 2;
        mu = max(.99 * mu, 0);
        %mu = max(mu, 0);
        Z = X - L - S;
        Z(unobserved) = 0;
        res = norm(Z, 'fro') / norm(X, 'fro');
        
    end

end

function r = So(tau, X)
    % shrinkage operator
    r = sign(X) .* max(abs(X) - tau, 0);
end

function r = Do(tau, X)
    % shrinkage operator for singular values
    [U, S, V] = svd(X, 'econ');
    r = U*So(tau, S)*V';
end

