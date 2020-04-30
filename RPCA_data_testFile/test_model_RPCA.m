clear; close all;

%% test on the random data

% load random_data;
fprintf('Penalty Solver:\n');
fprintf('Random Data Test\n');
S = load('random_data.mat');
X = S.X;

% Lopt and Sopt are the optimal low-rank and sparse matrices
Lopt = S.Lopt;
Sopt = S.Sopt;

% call your solver to obtain (L, S)
tic;
[X, L, S, Y, Z, res, iter] = penalty_solver(X);
toc;
fprintf('||L-Lopt||/||Lopt|| = %5.4e\n',norm(L-Lopt,'fro')/norm(Lopt,'fro'));
fprintf('||S-Sopt||/||Sopt|| = %5.4e\n\n',norm(S-Sopt,'fro')/norm(Sopt,'fro'));
fprintf('Elapsed Time: %f seconds\n', toc);
fprintf('Iterations: %d\n', iter);
fprintf('Final residual: %5.4e\n', res);


fprintf('--------------------\n');
%% test on the escalator data
fprintf('Escalator Data Test\n');
escalator = load('escalator_data.mat');
M = escalator.M;
% load escalator_data;
[m,n,p] = size(M);

X = zeros(m*n,p);
for i = 1:p
    X(:,i) = reshape(M(:,:,i),m*n,1);
end

% call your solver to obtain (L, S)
tic;
[X, L, S, Y, Z, res, iter] = penalty_solver(X);
toc;
fprintf('See Output Images\n')
fprintf('Elapsed Time: %f seconds\n', toc);
fprintf('Iterations: %d\n', iter);
fprintf('Final residual: %5.4e\n', res);

% show a few slices

L3D = zeros(size(M)); % 3-d format
S3D = zeros(size(M)); % 3-d format

for i = 1:p
    L3D(:,:,i) = reshape(L(:,i),m,n);
    S3D(:,:,i) = reshape(S(:,i),m,n);
end

fig = figure('papersize',[15,4],'paperposition',[0,0,15,4]);
subplot(1,3,1);
imshow(L3D(:,:,1),[]);
subplot(1,3,2);
imshow(L3D(:,:,100),[]);
subplot(1,3,3);
imshow(L3D(:,:,200),[]);
print(fig,'-dpdf','L3slices_Penalty');

fig = figure('papersize',[15,4],'paperposition',[0,0,15,4]);
subplot(1,3,1);
imshow(S3D(:,:,1),[]);
subplot(1,3,2);
imshow(S3D(:,:,100),[]);
subplot(1,3,3);
imshow(S3D(:,:,200),[]);
print(fig,'-dpdf','S3slices_Penalty');

%% Run the ADMM Solver
% load random_data;
fprintf('ADMM Solver:\n');
fprintf('Random Data Test\n');
S = load('random_data.mat');
X = S.X;

% Lopt and Sopt are the optimal low-rank and sparse matrices
Lopt = S.Lopt;
Sopt = S.Sopt;

tic;
[X, L, S, Y, Z, res, iter] = admm_solver(X);
toc;
fprintf('||L-Lopt||/||Lopt|| = %5.4e\n',norm(L-Lopt,'fro')/norm(Lopt,'fro'));
fprintf('||S-Sopt||/||Sopt|| = %5.4e\n\n',norm(S-Sopt,'fro')/norm(Sopt,'fro'));
fprintf('Elapsed Time: %f seconds\n', toc);
fprintf('Iterations: %d\n', iter);
fprintf('Final residual: %5.4e\n', res);


fprintf('--------------------\n');
%% test on the escalator data
fprintf('Escalator Data Test\n');
escalator = load('escalator_data.mat');
M = escalator.M;
% load escalator_data;
[m,n,p] = size(M);

X = zeros(m*n,p);
for i = 1:p
    X(:,i) = reshape(M(:,:,i),m*n,1);
end

% call your solver to obtain (L, S)
tic;
[X, L, S, Y, Z, res, iter] = admm_solver(X);
toc;
fprintf('See Output Images\n')
fprintf('Elapsed Time: %f seconds\n', toc);
fprintf('Iterations: %d\n', iter);
fprintf('Final residual: %5.4e\n', res);

% show a few slices

L3D = zeros(size(M)); % 3-d format
S3D = zeros(size(M)); % 3-d format

for i = 1:p
    L3D(:,:,i) = reshape(L(:,i),m,n);
    S3D(:,:,i) = reshape(S(:,i),m,n);
end

fig = figure('papersize',[15,4],'paperposition',[0,0,15,4]);
subplot(1,3,1);
imshow(L3D(:,:,1),[]);
subplot(1,3,2);
imshow(L3D(:,:,100),[]);
subplot(1,3,3);
imshow(L3D(:,:,200),[]);
print(fig,'-dpdf','L3slices_ADMM');

fig = figure('papersize',[15,4],'paperposition',[0,0,15,4]);
subplot(1,3,1);
imshow(S3D(:,:,1),[]);
subplot(1,3,2);
imshow(S3D(:,:,100),[]);
subplot(1,3,3);
imshow(S3D(:,:,200),[]);
print(fig,'-dpdf','S3slices_ADMM');
