%% Multi-Subject fMRI Analysis Pipeline (MATLAB)
% This script performs group-level analysis across 2 subjects
% Replicates Python multi-subject analysis from Assignment 2

clear; close all; clc;

%% ===== CONFIGURATION =====
subject_ids = [1, 2];  % Analyzing 2 subjects
hrf_path = 'hrf.mat';

fprintf('=== Multi-Subject Analysis (%d subjects) ===\n', length(subject_ids));

%% ===== INITIALIZE STORAGE =====
% We'll store results for each subject and then aggregate
all_subjects_roi_psc = struct();
condition_names = {};

%% ===== LOOP THROUGH SUBJECTS =====
for s = 1:length(subject_ids)
    subject_id = subject_ids(s);
    fprintf('\n--- Processing Subject %d ---\n', subject_id);

    % Define paths
    bold_path = sprintf('subj%d/bold.nii.gz', subject_id);
    labels_path = sprintf('subj%d/labels.txt', subject_id);
    mask_vt_path = sprintf('subj%d/mask4_vt.nii.gz', subject_id);
    mask_face_path = sprintf('subj%d/mask8_face_vt.nii.gz', subject_id);
    mask_house_path = sprintf('subj%d/mask8_house_vt.nii.gz', subject_id);

    %% Load data
    fprintf('Loading BOLD data...\n');
    bold_img = niftiread(bold_path);
    [X, Y, Z, T] = size(bold_img);

    fprintf('Loading labels...\n');
    labels = readtable(labels_path, 'Delimiter', ' ');
    labels.Properties.VariableNames = {'Condition', 'Run'};

    %% Create design matrix
    fprintf('Creating design matrix...\n');
    [design_matrix, condition_names_subj] = create_design_matrix(labels);
    design_matrix_with_intercepts = add_run_intercepts(design_matrix, labels);

    % Store condition names (same across subjects)
    if isempty(condition_names)
        condition_names = condition_names_subj;
    end

    %% Convolve with HRF
    fprintf('Convolving with HRF...\n');
    hrf_data = load(hrf_path);
    hrf_sampled = hrf_data.hrf_sampled;
    convolved_matrix = convolve_design_matrix(design_matrix_with_intercepts, ...
        hrf_sampled, condition_names);

    %% Fit GLM
    fprintf('Fitting GLM...\n');
    [beta_maps, residuals] = fit_glm(bold_img, convolved_matrix);
    df = T - rank(convolved_matrix);

    %% Load ROI masks
    fprintf('Loading ROI masks...\n');
    mask_vt = niftiread(mask_vt_path) > 0;
    mask_face = niftiread(mask_face_path) > 0;
    mask_house = niftiread(mask_house_path) > 0;

    roi_masks = struct();
    roi_masks.VentralTemporal = mask_vt;
    roi_masks.Face = mask_face;
    roi_masks.House = mask_house;

    %% ROI analysis
    fprintf('Performing ROI analysis...\n');
    [~, roi_mean_psc, ~] = compute_roi_percent_signal_change(...
        beta_maps, roi_masks, condition_names);

    %% Store results for this subject
    roi_names = fieldnames(roi_mean_psc);
    for r = 1:length(roi_names)
        roi_name = roi_names{r};

        if ~isfield(all_subjects_roi_psc, roi_name)
            all_subjects_roi_psc.(roi_name) = [];
        end

        % Append this subject's data
        all_subjects_roi_psc.(roi_name) = [all_subjects_roi_psc.(roi_name); ...
                                             roi_mean_psc.(roi_name)];
    end

    fprintf('Subject %d complete\n', subject_id);
end

%% ===== GROUP-LEVEL AGGREGATION =====
fprintf('\n=== Computing Group Statistics ===\n');

group_mean_psc = struct();
group_sem_psc = struct();

for r = 1:length(roi_names)
    roi_name = roi_names{r};
    subject_data = all_subjects_roi_psc.(roi_name);  % [N_subjects x N_conditions]

    % Compute mean and SEM across subjects
    group_mean_psc.(roi_name) = mean(subject_data, 1);  % Mean across subjects
    group_sem_psc.(roi_name) = std(subject_data, 0, 1) / sqrt(size(subject_data, 1));  % SEM

    fprintf('ROI: %s\n', roi_name);
    fprintf('  Mean PSC: ');
    fprintf('%.2f ', group_mean_psc.(roi_name));
    fprintf('\n');
end

%% ===== VISUALIZATION =====
fprintf('\nVisualizing group results...\n');
plot_roi_results(group_mean_psc, group_sem_psc, condition_names);

%% ===== SAVE GROUP RESULTS =====
fprintf('Saving group results...\n');
group_results = struct();
group_results.subject_ids = subject_ids;
group_results.condition_names = condition_names;
group_results.group_mean_psc = group_mean_psc;
group_results.group_sem_psc = group_sem_psc;
group_results.all_subjects_roi_psc = all_subjects_roi_psc;

save('group_results_2subjects.mat', 'group_results');

fprintf('\n=== Multi-Subject Analysis Complete ===\n');
fprintf('Results saved to: group_results_2subjects.mat\n');
