function design_matrix_with_intercepts = add_run_intercepts(design_matrix, labels)
% ADD_RUN_INTERCEPTS Adds intercept columns for each run
%
% Inputs:
%   design_matrix - Original design matrix [T x N_conditions]
%   labels - Table with 'Condition' and 'Run' columns
%
% Outputs:
%   design_matrix_with_intercepts - Design matrix with run intercepts

    % Get unique runs
    unique_runs = unique(labels.Run);
    n_runs = length(unique_runs);

    % Initialize intercept matrix
    n_timepoints = size(design_matrix, 1);
    intercepts = zeros(n_timepoints, n_runs);

    % Create intercept columns (one per run)
    for i = 1:n_runs
        run_id = unique_runs(i);
        intercepts(:, i) = (labels.Run == run_id);
    end

    % Concatenate design matrix with intercepts
    design_matrix_with_intercepts = [design_matrix, intercepts];

    fprintf('Added %d run intercepts\n', n_runs);
end
