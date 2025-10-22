function [roi_psc, roi_mean_psc, roi_sem] = compute_roi_percent_signal_change(...
    beta_maps, roi_masks, condition_names)
% COMPUTE_ROI_PERCENT_SIGNAL_CHANGE Computes percent signal change for ROIs
%
% Inputs:
%   beta_maps - Beta coefficients [X x Y x Z x N_regressors]
%   roi_masks - Struct with ROI masks (logical arrays)
%   condition_names - Cell array of condition names
%
% Outputs:
%   roi_psc - Cell array of PSC for each ROI
%   roi_mean_psc - Struct with mean PSC per ROI
%   roi_sem - Struct with SEM per ROI

    fprintf('Computing ROI percent signal change...\n');

    % Get dimensions
    [X, Y, Z, n_regressors] = size(beta_maps);
    n_conditions = length(condition_names);

    % Last 12 columns are intercepts (baseline for each run)
    n_runs = 12;
    intercept_indices = (n_regressors - n_runs + 1):n_regressors;
    condition_indices = 1:n_conditions;

    % Initialize outputs
    roi_names = fieldnames(roi_masks);
    roi_psc = struct();
    roi_mean_psc = struct();
    roi_sem = struct();

    % Process each ROI
    for r = 1:length(roi_names)
        roi_name = roi_names{r};
        mask = roi_masks.(roi_name);

        % Extract beta values for this ROI
        beta_2d = reshape(beta_maps, X*Y*Z, n_regressors);  % [N_voxels x N_regressors]
        mask_1d = mask(:);  % Flatten mask
        roi_betas = beta_2d(mask_1d, :);  % [N_roi_voxels x N_regressors]

        if isempty(roi_betas)
            fprintf('Warning: ROI %s has no voxels. Skipping.\n', roi_name);
            continue;
        end

        % Extract condition betas and intercepts
        condition_betas = roi_betas(:, condition_indices);
        intercept_betas = roi_betas(:, intercept_indices);

        % Compute mean intercept (baseline) for each voxel
        mean_baseline = mean(intercept_betas, 2);  % [N_roi_voxels x 1]

        % Compute percent signal change: (beta / baseline) * 100
        psc = (condition_betas ./ mean_baseline) * 100;  % [N_roi_voxels x N_conditions]

        % Store results
        roi_psc.(roi_name) = psc;
        roi_mean_psc.(roi_name) = mean(psc, 1);  % Mean across voxels
        roi_sem.(roi_name) = std(psc, 0, 1) / sqrt(size(psc, 1));  % SEM

        fprintf('  %s: %d voxels\n', roi_name, size(psc, 1));
    end

    fprintf('ROI analysis complete\n');
end
