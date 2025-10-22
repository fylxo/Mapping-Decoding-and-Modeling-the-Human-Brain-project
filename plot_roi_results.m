function plot_roi_results(roi_mean_psc, roi_sem, condition_names)
% PLOT_ROI_RESULTS Plots ROI percent signal change as bar graphs
%
% Inputs:
%   roi_mean_psc - Struct with mean PSC per ROI
%   roi_sem - Struct with SEM per ROI
%   condition_names - Cell array of condition names

    roi_names = fieldnames(roi_mean_psc);
    n_rois = length(roi_names);
    n_conditions = length(condition_names);

    % Create color map
    colors = lines(n_conditions);

    % Create figure with subplots
    n_cols = 2;
    n_rows = ceil(n_rois / n_cols);

    figure('Name', 'ROI Analysis', 'Position', [100 100 1000 400*n_rows]);

    for r = 1:n_rois
        subplot(n_rows, n_cols, r);

        roi_name = roi_names{r};
        mean_vals = roi_mean_psc.(roi_name);
        sem_vals = roi_sem.(roi_name);

        % Create bar plot with error bars
        b = bar(1:n_conditions, mean_vals);
        b.FaceColor = 'flat';
        b.CData = colors;

        hold on;
        errorbar(1:n_conditions, mean_vals, sem_vals, 'k.', 'LineWidth', 1.5);
        hold off;

        % Formatting
        title(strrep(roi_name, '_', ' '), 'FontSize', 12);
        xlabel('Conditions', 'FontSize', 10);
        ylabel('Percent Signal Change (%)', 'FontSize', 10);
        set(gca, 'XTick', 1:n_conditions, 'XTickLabel', condition_names, ...
            'XTickLabelRotation', 45);
        ylim([-1.5 max([roi_mean_psc.(roi_names{:})]) * 1.1]);
        grid on;
    end

    % Add overall title
    sgtitle('ROI Percent Signal Change', 'FontSize', 14, 'FontWeight', 'bold');
end
