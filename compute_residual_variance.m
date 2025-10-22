function residual_variance = compute_residual_variance(residuals, df, bold_shape)
% COMPUTE_RESIDUAL_VARIANCE Computes voxelwise residual variance
%
% Inputs:
%   residuals - Residuals [T x N_voxels]
%   df - Degrees of freedom
%   bold_shape - Original BOLD shape [X Y Z T]
%
% Outputs:
%   residual_variance - Variance map [X x Y x Z]

    % Compute voxelwise sum of squared residuals
    ss_residuals = sum(residuals.^2, 1);  % [1 x N_voxels]

    % Divide by degrees of freedom
    variance = ss_residuals / df;

    % Reshape to 3D
    X = bold_shape(1);
    Y = bold_shape(2);
    Z = bold_shape(3);
    residual_variance = reshape(variance, X, Y, Z);

    fprintf('Residual variance computed\n');
end
