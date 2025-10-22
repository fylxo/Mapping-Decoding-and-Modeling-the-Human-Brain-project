function convolved_matrix = convolve_design_matrix(design_matrix, hrf_sampled, condition_names)
% CONVOLVE_DESIGN_MATRIX Convolves condition columns with HRF
%
% Inputs:
%   design_matrix - Design matrix with conditions and intercepts
%   hrf_sampled - Sampled HRF vector
%   condition_names - Cell array of condition names
%
% Outputs:
%   convolved_matrix - Design matrix with convolved conditions

    n_conditions = length(condition_names);
    n_timepoints = size(design_matrix, 1);
    n_total_regressors = size(design_matrix, 2);

    % Initialize convolved matrix
    convolved_matrix = zeros(n_timepoints, n_total_regressors);

    % Convolve only the condition columns (not intercepts)
    for i = 1:n_conditions
        % Convolve with HRF
        convolved = conv(design_matrix(:, i), hrf_sampled, 'full');

        % Truncate to original length
        convolved_matrix(:, i) = convolved(1:n_timepoints);
    end

    % Copy intercepts without modification
    intercept_start = n_conditions + 1;
    convolved_matrix(:, intercept_start:end) = design_matrix(:, intercept_start:end);

    fprintf('Convolved %d conditions with HRF\n', n_conditions);
end
