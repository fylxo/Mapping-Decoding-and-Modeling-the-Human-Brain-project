%% Single Subject fMRI Analysis Pipeline (MATLAB)
% This script replicates the Python analysis from Assignment 1 & 2
% Author: Your Name
% Date: 2025

clear; close all; clc;

%% ===== CONFIGURATION =====
subject_id = 1;  % Change to 2 for second subject
bold_path = sprintf('../subj%d/bold.nii.gz', subject_id);
labels_path = sprintf('../subj%d/labels.txt', subject_id);
mask_vt_path = sprintf('../subj%d/mask4_vt.nii.gz', subject_id);
mask_face_path = sprintf('../subj%d/mask8_face_vt.nii.gz', subject_id);
mask_house_path = sprintf('../subj%d/mask8_house_vt.nii.gz', subject_id);
hrf_path = '../hrf.mat';

fprintf('=== Analyzing Subject %d ===\n', subject_id);

%% ===== PART 1: DATA LOADING =====
fprintf('Loading BOLD data...\n');
bold_img = niftiread(bold_path);
bold_info = niftiinfo(bold_path);

% Get dimensions
[X, Y, Z, T] = size(bold_img);
fprintf('BOLD data shape: [%d, %d, %d, %d]\n', X, Y, Z, T);

% Load labels
fprintf('Loading labels...\n');
labels = readtable(labels_path, 'Delimiter', ' ');
labels.Properties.VariableNames = {'Condition', 'Run'};

% Display unique conditions
unique_conditions = unique(labels.Condition);
fprintf('Unique conditions: ');
fprintf('%s ', unique_conditions{:});
fprintf('\n');

%% ===== PART 2: VISUALIZE SLICES =====
fprintf('Visualizing anatomical slices...\n');
mid_volume = round(T/2);
mid_z = round(Z/2);

figure('Name', 'Anatomical Slices', 'Position', [100 100 1200 400]);

% Axial slice
subplot(1,3,1);
imagesc(squeeze(bold_img(:,:,mid_z,mid_volume))');
colormap gray; colorbar;
title(sprintf('Axial Slice (z=%d)', mid_z));
xlabel('Left ↔ Right'); ylabel('Anterior ↔ Posterior');
axis image;

% Coronal slice
subplot(1,3,2);
mid_y = round(Y/2);
imagesc(squeeze(bold_img(:,mid_y,:,mid_volume))');
colormap gray; colorbar;
title(sprintf('Coronal Slice (y=%d)', mid_y));
xlabel('Left ↔ Right'); ylabel('Superior ↔ Inferior');
axis image;

% Sagittal slice
subplot(1,3,3);
mid_x = round(X/2);
imagesc(squeeze(bold_img(mid_x,:,:,mid_volume))');
colormap gray; colorbar;
title(sprintf('Sagittal Slice (x=%d)', mid_x));
xlabel('Anterior ↔ Posterior'); ylabel('Superior ↔ Inferior');
axis image;

%% ===== PART 3: DESIGN MATRIX =====
fprintf('Creating design matrix...\n');
[design_matrix, condition_names] = create_design_matrix(labels);
fprintf('Design matrix shape: [%d, %d]\n', size(design_matrix));

% Add run-specific intercepts
design_matrix_with_intercepts = add_run_intercepts(design_matrix, labels);
fprintf('Design matrix with intercepts: [%d, %d]\n', size(design_matrix_with_intercepts));

% Visualize design matrix
figure('Name', 'Design Matrix');
imagesc(design_matrix_with_intercepts);
colormap gray; colorbar;
title('Design Matrix with Run Intercepts');
xlabel('Regressors'); ylabel('Time (Volumes)');

%% ===== PART 4: HRF CONVOLUTION =====
fprintf('Loading HRF and convolving...\n');
hrf_data = load(hrf_path);
hrf_sampled = hrf_data.hrf_sampled;

% Visualize HRF
figure('Name', 'HRF');
plot(hrf_sampled, 'LineWidth', 2);
title('Hemodynamic Response Function (HRF)');
xlabel('Time (samples)'); ylabel('Amplitude');
grid on;

% Convolve design matrix
convolved_matrix = convolve_design_matrix(design_matrix_with_intercepts, ...
    hrf_sampled, condition_names);
fprintf('Convolved matrix shape: [%d, %d]\n', size(convolved_matrix));

% Visualize convolved matrix
figure('Name', 'Convolved Design Matrix');
imagesc(convolved_matrix);
colormap gray; colorbar;
title('Convolved Design Matrix');
xlabel('Regressors'); ylabel('Time (Volumes)');

%% ===== PART 5: FIT GLM =====
fprintf('Fitting GLM...\n');
[beta_maps, residuals] = fit_glm(bold_img, convolved_matrix);
fprintf('Beta maps shape: [%d, %d, %d, %d]\n', size(beta_maps));

% Calculate degrees of freedom
df = T - rank(convolved_matrix);
fprintf('Degrees of freedom: %d\n', df);

%% ===== PART 6: COMPUTE T-MAPS =====
fprintf('Computing T-maps...\n');
t_maps = compute_t_maps(beta_maps, convolved_matrix, residuals, df);
fprintf('T-maps shape: [%d, %d, %d, %d]\n', size(t_maps));

% Visualize T-map for a specific condition
condition_to_plot = 'face';
condition_idx = find(strcmp(condition_names, condition_to_plot));
slice_to_plot = 26;

figure('Name', sprintf('T-map: %s', condition_to_plot));
t_slice = squeeze(t_maps(:,:,slice_to_plot,condition_idx));
imagesc(t_slice'); clim([-10 10]);
colormap(redblue); colorbar;
title(sprintf('T-map for %s (slice %d)', condition_to_plot, slice_to_plot));
xlabel('X'); ylabel('Y');
axis image;

%% ===== PART 7: CONTRAST MAPS =====
fprintf('Computing contrast maps...\n');

% Define contrast: house > face
contrast_vector = zeros(size(convolved_matrix, 2), 1);
house_idx = find(strcmp(condition_names, 'house'));
face_idx = find(strcmp(condition_names, 'face'));
contrast_vector(house_idx) = 1;
contrast_vector(face_idx) = -1;

% Compute residual variance
residual_variance = compute_residual_variance(residuals, df, size(bold_img));

% Compute contrast map
t_contrast = compute_contrast_map(beta_maps, convolved_matrix, ...
    contrast_vector, residual_variance, df);

% Visualize contrast map
figure('Name', 'Contrast: House > Face');
imagesc(squeeze(t_contrast(:,:,28))'); clim([-10 10]);
colormap(redblue); colorbar;
title('Contrast Map: House > Face (slice 28)');
xlabel('X'); ylabel('Y');
axis image;

%% ===== PART 8: ROI ANALYSIS =====
fprintf('Performing ROI analysis...\n');

% Load ROI masks
mask_vt = niftiread(mask_vt_path) > 0;
mask_face = niftiread(mask_face_path) > 0;
mask_house = niftiread(mask_house_path) > 0;

roi_masks = struct();
roi_masks.VentralTemporal = mask_vt;
roi_masks.Face = mask_face;
roi_masks.House = mask_house;

% Extract ROI percent signal change
[roi_psc, roi_mean_psc, roi_sem] = compute_roi_percent_signal_change(...
    beta_maps, roi_masks, condition_names);

% Plot ROI results
plot_roi_results(roi_mean_psc, roi_sem, condition_names);

%% ===== PART 9: SAVE RESULTS =====
fprintf('Saving results...\n');
results = struct();
results.subject_id = subject_id;
results.beta_maps = beta_maps;
results.t_maps = t_maps;
results.roi_mean_psc = roi_mean_psc;
results.roi_sem = roi_sem;
results.condition_names = condition_names;

save(sprintf('results_subj%d.mat', subject_id), 'results');

fprintf('=== Analysis Complete for Subject %d ===\n', subject_id);
