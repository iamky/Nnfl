% lab3_xor_multilayer.m
% Lab Session 3: Multilayer Feedforward Network using Backpropagation (2-input XOR)
% MATLAB version of a 2-8-4-1 MLP with ReLU (poslin) and Sigmoid (logsig).
% Loss: Binary Cross-Entropy, Training epochs: 1000

clear; clc; rng(42);

% ---------------------
% Data: XOR truth table
% ---------------------
% X (features x samples), Y (outputs x samples)
X = [0 0 1 1;   % x1
     0 1 0 1];  % x2
Y = [0 1 1 0];  % XOR labels

% --------------------------------------------
% Model: 2 -> 8 -> 4 -> 1 (Dense MLP)
% --------------------------------------------
% Two hidden layers: [8 4]
net = feedforwardnet([8 4], 'trainscg');

% Activations
net.layers[1].transferFcn = 'poslin';   % Hidden 1: ReLU
net.layers[2].transferFcn = 'poslin';   % Hidden 2: ReLU
net.layers[3].transferFcn = 'logsig';   % Output : Sigmoid

% Loss
net.performFcn = 'crossentropy';

% Train on all samples (no val/test split)
net.divideFcn = 'dividetrain';

% Training params
net.trainParam.epochs = 1000;
net.trainParam.showWindow = false;  % suppress GUI

% ---------------------
% Train
% ---------------------
[net, tr] = train(net, X, Y);

% ---------------------
% Predict & Evaluate
% ---------------------
Yhat = net(X);        % probabilities in (0,1)
Ypred = round(Yhat);  % 0/1

disp('Multilayer Feedforward XOR Output (rounded predictions):');
disp(Ypred');

acc = mean(Ypred == Y);
fprintf('Training Accuracy: %.2f%%\n', acc*100);

disp('Probabilities (Yhat):');
disp(Yhat');
