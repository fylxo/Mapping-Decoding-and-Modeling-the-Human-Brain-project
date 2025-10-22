function t_contrast = compute_contrast_map(beta_maps, design_matrix, ...
    contrast_vector, residual_variance, df)
% COMPUTE_CONTRAST_MAP Computes a contrast map with t-values
%
% Inputs:
%   beta_maps - Beta coefficients [X x Y x Z x N_regressors]
%   design_matrix - Design matrix [T x N_regressors]
%   contrast_vector - Contrast weights [N_regressors x 1]
%   residual_variance - Variance map [X x Y x Z]
%   df - Degrees of freedom
%
% Outputs:
%   t_contrast - T-values for the contrast [X x Y x Z]

    fprintf('Computing contrast map...\n');

    % Get dimensions
    [X, Y, Z, n_regressors] = size(beta_maps);

    % Ensure contrast vector is a column vector
    contrast_vector = contrast_vector(:);

    % Compute contrast variance: c' * (X'X)^-1 * c
    X_pinv = pinv(design_matrix);
    contrast_var_coeff = contrast_vector' * (X_pinv * X_pinv') * contrast_vector;

    % Compute contrast beta: c' * beta
    beta_2d = reshape(beta_maps, X*Y*Z, n_regressors)';  % [N_regressors x N_voxels]
    contrast_beta = contrast_vector' * beta_2d;  % [1 x N_voxels]

    % Reshape to 3D
    contrast_beta_3d = reshape(contrast_beta, X, Y, Z);

    % Compute standard error: SE = sqrt(sigma^2 * c'(X'X)^-1c)
    variance_threshold = 1e-6;
    residual_variance = max(residual_variance, variance_threshold);
    SE = sqrt(residual_variance * contrast_var_coeff);

    % Compute t-values: t = c'beta / SE
    t_contrast = contrast_beta_3d ./ SE;

    fprintf('Contrast map computation complete\n');
end
