function [X, L, S, Y, Z, err, iter] = alm_solver(X)

    [r, c] = size(X);
    unobserved = isnan(X);
    lambda = 1 / sqrt(max(r, c));
    mu = 10 * lambda;
    tol = 1e-4;
    normX = norm(X, 'fro');
    L = zeros(r, c);
    LNot = zeros(r, c);
    S = zeros(r, c);
    SNot = zeros(r, c);
    Y = zeros(r, c);
    k = 0;
    t = 1;
    tNot = 1;
    
    iter = 0;
    err = 1;
    err2 = 1e-7;
    stoppingCrit = 1e-7;
    while err > tol
        iter = iter + 1
        
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
        err = norm(Z, 'fro') / normX
        %err2 = min(mu, sqrt(mu)) * norm((S - SNot), 'fro') / normX;
        %stoppingCritL = 2 * (YL - L) + (L + S - YL - YS);
        %stoppingCritS = 2 * (YS - S) + (L + S - YL - YS);
        %stoppingCrit = sqrt(norm(stoppingCritL, 'fro')^2 + norm(stoppingCritS, 'fro')^2)
        
        
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

