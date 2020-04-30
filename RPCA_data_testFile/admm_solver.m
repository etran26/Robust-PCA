function [X, L, S, Y, Z, res, iter] = admm_solver(X)

    [r, c] = size(X);
    unobserved = isnan(X);
    X(unobserved) = 0;
    lambda = 1 / sqrt(max(r, c));
    mu = 10 * lambda;
    tol = 1e-4;
    L = zeros(r, c);
    S = zeros(r, c);
    Y = zeros(r, c);

    iter = 0;
    res = 1;
    while res > tol
        iter = iter + 1;
        L = Do(1 / mu, X - S + (1 / mu) * Y);
        S = So(lambda / mu, X - L + (1 / mu) * Y);
        Z = X - L - S;
        Z(unobserved) = 0;
        Y = Y + mu * Z;
        res = norm(Z, 'fro') / norm(X, 'fro');
        % mu = max(eta * mu, mu_bar);
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
