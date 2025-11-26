% xor_backprop_xor.m
% Lab Session 2: Backpropagation Learning Algorithm (2-input XOR)
% MATLAB version of a 2-4-1 MLP with ReLU (poslin) and Sigmoid (logsig).
% Loss: Binary Cross-Entropy, Training: 1000 epochs.

clear; clc; rng(42);

% XOR input (2 x N) and target (1 x N) -- columns are samples
X = [0 0 1 1;   % x1
     0 1 0 1];  % x2

Y = [0 1 1 0];  % XOR labels

% ---- Model: 2 -> 4 -> 1 ----
% Use feedforwardnet with 4 hidden neurons
% Trainer: 'trainscg' (scaled conjugate gradient) - stable for tiny problems
net = feedforwardnet(4, 'trainscg');

% Activations: ReLU for hidden (poslin), Sigmoid for output (logsig)
net.layers{1}.transferFcn = 'poslin';   % ReLU
net.layers{2}.transferFcn = 'logsig';   % Sigmoid

% Performance (loss): binary cross-entropy
net.performFcn = 'crossentropy';

% Tiny dataset: train on all samples (no val/test split)
net.divideFcn = 'dividetrain';

% Training params (epochs ~ iterations)
net.trainParam.epochs = 1000;
net.trainParam.showWindow = false;   % suppress GUI training window

% Train
[net, tr] = train(net, X, Y);

% Predict
Yhat = net(X);                % probabilities in (0,1)
Ypred = round(Yhat);          % 0/1 decisions

% Report
disp('XOR Output (rounded predictions):');
disp(Ypred');

acc = mean(Ypred == Y);
fprintf('Accuracy: %.2f%%\n', acc*100);

% (Optional) show raw probabilities
disp('Probabilities:');
disp(Yhat');
