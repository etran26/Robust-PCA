function [X, L, S, Y, Z, err, iter] = admm_solver(X)

    [r, c] = size(X);
    unobserved = isnan(X);
    X(unobserved) = 0;
    normX = norm(X, 'fro');
    lambda = 1 / sqrt(max(r, c));
    mu = 10 * lambda;
    %tol = 1e-6;
    tol = 1e-4;
    L = zeros(r, c);
    S = zeros(r, c);
    Y = zeros(r, c);

    iter = 0;
    %err = 1e-7;
    err = 1;
    while err > tol %err < tol
        iter = iter + 1
        L = Do(1 / mu, X - S + (1 / mu) * Y);
        S = So(lambda / mu, X - L + (1 / mu) * Y);
        Z = X - L - S;
        Z(unobserved) = 0;
        Y = Y + mu * Z;
        err = norm(Z, 'fro') / normX
        
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
