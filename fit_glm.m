function [beta_maps, residuals] = fit_glm(bold_img, design_matrix)
% FIT_GLM Fits a General Linear Model voxel-by-voxel
%
% Inputs:
%   bold_img - 4D BOLD data [X x Y x Z x T]
%   design_matrix - Design matrix [T x N_regressors]
%
% Outputs:
%   beta_maps - Beta coefficients [X x Y x Z x N_regressors]
%   residuals - Residuals [T x N_voxels]

    fprintf('Fitting GLM...\n');

    % Get dimensions
    [X, Y, Z, T] = size(bold_img);
    n_regressors = size(design_matrix, 2);

    % Reshape BOLD data to 2D: [T x Voxels]
    bold_2d = reshape(bold_img, X*Y*Z, T)';

    % Solve GLM: beta = (X'X)^-1 X'Y
    % Using pinv for numerical stability
    X_pinv = pinv(design_matrix);
    beta_matrix = X_pinv * bold_2d;  % [N_regressors x N_voxels]

    % Compute residuals
    residuals = bold_2d - (design_matrix * beta_matrix);  % [T x N_voxels]

    % Reshape beta coefficients back to 4D brain space
    beta_maps = reshape(beta_matrix', X, Y, Z, n_regressors);

    fprintf('GLM fitting complete. Beta maps shape: [%d %d %d %d]\n', ...
        X, Y, Z, n_regressors);
end
