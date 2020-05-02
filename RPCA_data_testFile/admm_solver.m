function [X, L, S, Y, Z, res, iter, objs, cv] = admm_solver(X)
    % initialize variables
    [r, c] = size(X);
    unobserved = isnan(X);
    X(unobserved) = 0;
    lambda = 1 / sqrt(max(r, c));
    mu = 10 * lambda;
    tol = 1e-4;
    L = zeros(r, c);
    S = zeros(r, c);
    Y = zeros(r, c);
    objs = [Obj(L,S, lambda)];
    cv = [1];
    iter = 0;
    res = 1;
    % stopping condition
    while res > tol
        iter = iter + 1;
        % subproblems
        L = SOSingular(1 / mu, X - S + (1 / mu) * Y);
        S = SO(lambda / mu, X - L + (1 / mu) * Y);
        Z = X - L - S;
        Z(unobserved) = 0;
        Y = Y + mu * Z;
        % primal feasibility
        res = norm(Z, 'fro') / norm(X, 'fro');
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
