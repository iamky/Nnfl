
% Logic Gate Perceptron in MATLAB

clc;
clear;

% Dictionary of logic gate outputs
logic_gates = struct( ...
    'AND',  [0; 0; 0; 1], ...
    'OR',   [0; 1; 1; 1], ...
    'NAND', [1; 1; 1; 0], ...
    'NOR',  [1; 0; 0; 0] ...
);

% Prompt user to choose a gate
disp('Choose a logic gate: AND, OR, NAND, NOR');
gate = upper(input('Enter gate name: ', 's'));

% Check if gate is valid
if ~isfield(logic_gates, gate)
    disp('Invalid gate selected. Please choose from AND, OR, NAND, NOR.');
    return;
end

% Input combinations
X = [0 0; 0 1; 1 0; 1 1];
Y = logic_gates.(gate);  % Get correct output vector

% Hyperparameters
epochs = 10;
lr = 0.1;
weights = zeros(1, 2);
bias = 0;

fprintf('\nTraining Perceptron for %s Gate\n\n', gate);

% Step activation function
activation = @(x) double(x > 0);

% Training loop
for epoch = 1:epochs
    fprintf('Epoch %d\n', epoch);
    for i = 1:size(X, 1)
        z = dot(X(i, :), weights) + bias;
        y_pred = activation(z);
        error = Y(i) - y_pred;

        % Update weights and bias
        weights = weights + lr * error * X(i, :);
        bias = bias + lr * error;

        fprintf('Input: [%d %d], Predicted: %d, Actual: %d, Error: %d, Weights: [%0.2f %0.2f], Bias: %0.2f\n', ...
                X(i,1), X(i,2), y_pred, Y(i), error, weights(1), weights(2), bias);
    end
    disp('--------------------------------------------------');
end

% Final result
fprintf('\nFinal Weights: [%0.2f %0.2f]\n', weights(1), weights(2));
fprintf('Final Bias: %0.2f\n', bias);
