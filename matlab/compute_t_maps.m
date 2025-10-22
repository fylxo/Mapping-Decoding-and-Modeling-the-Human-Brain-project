function t_maps = compute_t_maps(beta_maps, design_matrix, residuals, df)
% COMPUTE_T_MAPS Computes voxelwise t-values for beta coefficients
%
% Inputs:
%   beta_maps - Beta coefficients [X x Y x Z x N_regressors]
%   design_matrix - Design matrix [T x N_regressors]
%   residuals - Residuals [T x N_voxels]
%   df - Degrees of freedom
%
% Outputs:
%   t_maps - T-values [X x Y x Z x N_regressors]

    fprintf('Computing t-maps...\n');

    % Get dimensions
    [X, Y, Z, n_regressors] = size(beta_maps);
    T = size(design_matrix, 1);

    % Compute covariance matrix diagonal
    X_pinv = pinv(design_matrix);
    cov_diag = diag(X_pinv * X_pinv');

    % Ensure no zeros in cov_diag
    epsilon = 1e-10;
    cov_diag = max(cov_diag, epsilon);

    % Compute residual variance for each voxel
    residual_var = sum(residuals.^2, 1) / df;  % [1 x N_voxels]
    residual_var = max(residual_var, 1e-12);  % Avoid division by zero

    % Reshape for computation
    beta_2d = reshape(beta_maps, X*Y*Z, n_regressors)';  % [N_regressors x N_voxels]

    % Compute standard errors: SE = sqrt(sigma^2 * diag(cov))
    SE = sqrt(residual_var .* cov_diag);  % [N_regressors x N_voxels]
    SE = max(SE, epsilon);  % Avoid division by zero

    % Compute t-values: t = beta / SE
    t_values = beta_2d ./ SE;  % [N_regressors x N_voxels]

    % Reshape back to 4D
    t_maps = reshape(t_values', X, Y, Z, n_regressors);

    fprintf('T-map computation complete\n');
end
