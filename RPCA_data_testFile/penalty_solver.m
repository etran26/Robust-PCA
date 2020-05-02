function [X, L, S, Y, Z, res, iter, objs, cv] = penalty_solver(X)
    % initialize variables
    [r, c] = size(X);
    unobserved = isnan(X);
    lambda = 1 / sqrt(max(r, c));
    eta = 0.9;
    mu = 0.99 * norm(X);
    mu_bar = 1e-9 * mu;
    tol = 1e-4;
    L = zeros(r, c);
    LNot = zeros(r, c);
    S = zeros(r, c);
    SNot = zeros(r, c);
    Y = zeros(r, c);
    t = 1;
    objs = [Obj(L,S, lambda)];
    cv = [1];
    tNot = 1;
    iter = 0;
    res = 1;
    err = 1;
    % stopping condition
    while (res > tol || err > tol) && mu_bar < mu
        iter = iter + 1;
        % subproblems (closed form)
        YL = L + ((tNot - 1) / t) * (L - LNot);
        YS = S + ((tNot - 1) / t) * (S - SNot);
        GL = YL - 0.5 * (YL + YS - X);
        LNot = L;
        SNot = S;
        L = SOSingular(mu / 2, GL);
        GS = YS - 0.5 * (YL + YS - X);
        S = SO((lambda * mu) / 2, GS);
        tNot = t;
        t = (1 + sqrt((4 * (t^2)) + 1)) / 2;
        % update mu
        mu = max(eta * mu, mu_bar);
        Z = X - L - S;
        Z(unobserved) = 0;
        % primal feasibility
        res = norm(Z, 'fro') / norm(X, 'fro');
        % dual feasibility
        err = max(mu, sqrt(mu)) * norm((S - SNot), 'fro') / norm(X, 'fro');
        % store objective and constraint violation for graphing
        objs = [objs; Obj(L,S,lambda)];
        cv = [cv; res];
    end

end

function r = SO(tau, X)
    % shrinkage operator
    r = sign(X) .* max(abs(X) - tau, 0);
end

function r = SOSingular(tau, X)
    % shrinkage operator (singular values case)
    [U, S, V] = svd(X, 'econ');
    r = U*SO(tau, S)*V';
end

function r = Obj(L, S, lambda)
    r = norm(svd(L),1) + lambda * norm(S,1);
end

