function [design_matrix, condition_names] = create_design_matrix(labels)
% CREATE_DESIGN_MATRIX Creates a design matrix from labels
%
% Inputs:
%   labels - Table with 'Condition' and 'Run' columns
%
% Outputs:
%   design_matrix - Binary design matrix [T x N_conditions]
%   condition_names - Cell array of condition names (excluding 'rest')

    % Get unique conditions
    all_conditions = unique(labels.Condition);

    % Remove 'rest' condition
    condition_names = all_conditions(~strcmp(all_conditions, 'rest'));

    % Initialize design matrix
    n_timepoints = height(labels);
    n_conditions = length(condition_names);
    design_matrix = zeros(n_timepoints, n_conditions);

    % Fill design matrix with binary indicators
    for i = 1:n_conditions
        condition = condition_names{i};
        design_matrix(:, i) = strcmp(labels.Condition, condition);
    end

    fprintf('Created design matrix: %d timepoints x %d conditions\n', ...
        n_timepoints, n_conditions);
end
