close all
clear
clc
 
load('Dist.mat');
 
euro_n = length(euro_label);
%us_n = length(us_label);
 
no_dims = 2;
 
triplets = [];
j = repmat(1:euro_n-1, 1, euro_n-1);
k = repmat(1:euro_n-1, euro_n-1, 1);
k = k(:)';
jk = [j(j<k); k(j<k)]';
len_jk = size(jk, 1);
for n = 1:euro_n
    temp = jk+(jk>=n);
    triplets = [triplets; [repmat(n, len_jk, 1), temp]];
end
temp = euro_dist(triplets(:, 1)+(triplets(:, 2)-1)*euro_n)-euro_dist(triplets(:, 1)+(triplets(:, 3)-1)*euro_n);
triplets = triplets(temp ~= 0, :);
temp = temp(temp ~= 0);
temp = find(temp > 0);
triplets(temp, [2; 3]) = triplets(temp, [3; 2]);
num_triplets = size(triplets, 1);
perm = randperm(num_triplets);
triplets = triplets(perm, :);
 
techniques = {'ProjGD', 'FGD', 'SFGD', 'SVRG', 'SVRG-BB', 'SVRG-SBB_0' 'SVRG-SBB_\epsilon'};
num_train = num_triplets;
 num_test = num_triplets;
alpha = no_dims-1;
delta = 0.05;
svrg_iter  = 120;
no_repeat = 1;
eta = 0.0002;
eta_par = 0.01;
scheduling = 1;
error_type = 1;
 
train_errors = zeros(length(techniques), 1+svrg_iter);
test_errors = zeros(length(techniques), 1+svrg_iter);
Predict_X = zeros(length(techniques), svrg_iter, euro_n, no_dims);
run_time = zeros(length(techniques), svrg_iter);
function_value = zeros(3, svrg_iter);
eta_seq = zeros(3, svrg_iter);
 
X_int = randn(euro_n, no_dims);
sum_X = sum(X_int .^ 2, 2);
D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X_int * X_int')));
 
train_triplets = triplets;
test_triplets = triplets;
labels = ones(euro_n, 1);
 
no_train_viol = sum(D(sub2ind([euro_n euro_n], train_triplets(:, 1), train_triplets(:, 2))) > ...
        D(sub2ind([euro_n euro_n], train_triplets(:, 1), train_triplets(:, 3))));
no_test_viol = sum(D(sub2ind([euro_n euro_n], test_triplets(:, 1), test_triplets(:, 2))) > ...
        D(sub2ind([euro_n euro_n], test_triplets(:, 1), test_triplets(:, 3))));
no_train = size(train_triplets, 1);
no_test = size(test_triplets, 1);
train_errors(:, 1) = no_train_viol/no_train;
test_errors(:, 1) = no_test_viol/no_test;
train_triplets_stoch = train_triplets - 1;
test_triplets_stoch = test_triplets - 1;
train_triplets_batch = train_triplets;
test_triplets_batch = test_triplets;
frq_iter = no_repeat*no_train;
sgd_iter = (no_repeat+2)*svrg_iter;
batch_iter = sgd_iter;
 
epsilon = 0.1;
 
Y = cmdscale(euro_dist, 2);
G = Y*Y';
D = bsxfun(@plus, bsxfun(@plus, -2 .* G, diag(G)), diag(G)');
row_mean = mean(D, 1);
column_mean = mean(D, 2);
total_mean = mean(row_mean);
ROW_MEAN = repmat(row_mean, euro_n, 1);
COLUMN_MEAN = repmat(column_mean, 1, euro_n);
D = D+total_mean-ROW_MEAN-COLUMN_MEAN;
[V, L] = eig(D);
Y = V(:,1:no_dims) * L(1:no_dims, 1:no_dims);
Y = normc(Y);
 
for k = 1:length(techniques)
    switch techniques{k}
        case 'SFGD'
            tic
            [Predict_X(k, :, :, :), train_errors(k, 2:end), test_errors(k, 2:end), run_time(k, :)] = gnmds_sgd_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, euro_n, no_dims, ...
                no_train, no_test, eta, eta_par, scheduling, sgd_iter, svrg_iter, error_type);
            toc
        case 'SVRG'
            tic
            [Predict_X(k, :, :, :), train_errors(k, 2:end), test_errors(k, 2:end), run_time(k, :)] = gnmds_svrg_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, euro_n, no_dims, ...
                no_train, no_test, 3*eta, frq_iter, svrg_iter, error_type);
            toc
        case 'FGD'
            tic
            [Predict_X(k, :, :, :), train_errors(k, 2:end), test_errors(k, 2:end), run_time(k, :)] = gnmds_x_epoch_time(X_int, train_triplets_batch, test_triplets_batch, no_dims, 30*eta, no_repeat, ...
                batch_iter, svrg_iter);
            toc
        case 'ProjGD'
            tic
            [Predict_X(k, :, :, :), train_errors(k, 2:end), test_errors(k, 2:end), run_time(k, :)] = gnmds_kernel(X_int, train_triplets_batch, test_triplets_batch, no_dims, eta, no_repeat, ...
                batch_iter, svrg_iter, 0);
            toc
        case 'SVRG-BB'
            tic
            [Predict_X(k, :, :, :), train_errors(k, 2:end), test_errors(k, 2:end), run_time(k, :), eta_seq(1, :)] = gnmds_svrg_bb_epsilon_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, euro_n, no_dims, ...
                no_train, no_test, 3*eta, 0, 1, frq_iter, svrg_iter, error_type);
        case 'SVRG-SBB_0'
            tic
            [Predict_X(k, :, :, :), train_errors(k, 2:end), test_errors(k, 2:end), run_time(k, :), eta_seq(2, :)] = gnmds_svrg_bb_epsilon_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, euro_n, no_dims, ...
                no_train, no_test, 3*eta, 0, 2, frq_iter, svrg_iter, error_type);
            toc
        case 'SVRG-SBB_\epsilon'
            tic
            [Predict_X(k, :, :, :), train_errors(k, 2:end), test_errors(k, 2:end), run_time(k, :), eta_seq(3, :)] = gnmds_svrg_bb_epsilon_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, euro_n, no_dims, ...
                no_train, no_test, 3*eta, epsilon, 2, frq_iter, svrg_iter, error_type);
            toc
    end
    K = squeeze(Predict_X(k, end, :, :))*squeeze(Predict_X(k, end, :, :))';
    G = K*K';
    D = bsxfun(@plus, bsxfun(@plus, -2 .* G, diag(G)), diag(G)');
    row_mean = mean(D, 1);
    column_mean = mean(D, 2);
    total_mean = mean(row_mean);
    ROW_MEAN = repmat(row_mean, euro_n, 1);
    COLUMN_MEAN = repmat(column_mean, 1, euro_n);
    D = D+total_mean-ROW_MEAN-COLUMN_MEAN;
    [V, L] = eig(D);
    X = V(:,1:no_dims) * L(1:no_dims, 1:no_dims);
    X = normc(X);
    figure;
    scatter(X(:, 1), X(:, 2), 'r', 'd');hold on
    scatter(Y(:, 1), Y(:, 2), 'b', 'filled', 's');hold on
    legend(techniques{k}, 'MDS');
    for i = 1:size(X,1)
        text(X(i, 1), X(i, 2), euro_label{i}, 'Color', 'r', 'FontAngle', 'italic');
    end
    for i = 1:size(Y,1)
        text(Y(i, 1), Y(i, 2), euro_label{i}, 'Color', 'b', 'FontWeight', 'bold');
    end
end
 
% lineColor = linspecer(length(techniques));
% x = 1:3:3*(svrg_iter+1);
% figure;
% plot(x(1:1:end), test_errors(1, 1:1:end), 'Color', lineColor(7, :), 'LineStyle', '-' , 'Marker', 'o', 'LineWidth', 3);hold on
% plot(x(1:1:end), test_errors(2, 1:1:end), 'Color', lineColor(6, :), 'LineStyle', '--', 'Marker', '*', 'LineWidth', 3);hold on
% plot(x(1:1:end), test_errors(3, 1:1:end), 'Color', lineColor(5, :), 'LineStyle', ':' , 'Marker', 'x', 'LineWidth', 3);hold on
% plot(x(1:1:end), test_errors(4, 1:1:end), 'Color', lineColor(4, :), 'LineStyle', '-.', 'Marker', '+', 'LineWidth', 3);hold on
% plot(x(1:1:end), test_errors(5, 1:1:end), 'Color', lineColor(3, :), 'LineStyle', '-' , 'Marker', '^', 'LineWidth', 3);hold on
% plot(x(1:1:end), test_errors(6, 1:1:end), 'Color', lineColor(2, :), 'LineStyle', '--', 'Marker', 'v', 'LineWidth', 3);hold on
% plot(x(1:1:end), test_errors(7, 1:1:end), 'Color', lineColor(1, :), 'LineStyle', '-.', 'Marker', 's', 'LineWidth', 3);hold on
% legend('ProjGD', 'FGD', 'SFGD', 'SVRG', 'SVRG-BB', 'SVRG-SBB_0', 'SVRG-SBB_\epsilon');
% xlabel('#gradients / #constraints');
% ylabel('Test Error');
% xlim([0 3*svrg_iter+10]);
% set(gca, 'FontName', 'Arial','FontSize', 16);
% set(findall(gcf,'type','text'), 'FontName', 'Arial', 'FontSize', 16);

lineColor = linspecer(length(techniques));
x = 1:3:3*(svrg_iter+1);
figure;
plot(x(1:1:end), test_errors(1, 1:1:end), 'Color', lineColor(7, :), 'LineStyle', '-' , 'Marker', 'o', 'LineWidth', 2);hold on
plot(x(1:1:end), test_errors(2, 1:1:end), 'Color', lineColor(6, :), 'LineStyle', '--', 'Marker', '*', 'LineWidth', 2);hold on
plot(x(1:1:end), test_errors(3, 1:1:end), 'Color', lineColor(5, :), 'LineStyle', ':' , 'Marker', 'x', 'LineWidth', 2);hold on
plot(x(1:1:end), test_errors(4, 1:1:end), 'Color', lineColor(4, :), 'LineStyle', '-.', 'Marker', '+', 'LineWidth', 2);hold on
% plot(x(1:1:end), test_errors(5, 1:1:end), 'Color', lineColor(3, :), 'LineStyle', '-' , 'Marker', '^', 'LineWidth', 3);hold on
plot(x(1:1:end), test_errors(6, 1:1:end), 'Color', lineColor(2, :), 'LineStyle', '--', 'Marker', 'v', 'LineWidth', 2);hold on
plot(x(1:1:end), test_errors(7, 1:1:end), 'Color', lineColor(1, :), 'LineStyle', '-.', 'Marker', 's', 'LineWidth', 2);hold on
legend('ProjGD (\eta=2e-4)', 'FGD (\eta=6e-3)', 'SFGD (\eta_k = 2e-4/k)', 'SVRG (\eta = 6e-4)', 'SVRG-SBB_0', 'SVRG-SBB_{0.1}');
xlabel('epoch number');
ylabel('Test Error');
axis([0,360,0,0.5]);
% xlim([0 3*svrg_iter+10]);
% set(gca, 'FontName', 'Arial','FontSize', 16);
% set(findall(gcf,'type','text'), 'FontName', 'Arial', 'FontSize', 16);
