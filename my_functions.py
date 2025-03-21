import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.io import loadmat
from numpy.linalg import lstsq
from scipy.stats import t


# ----------------------------- Project Work 1 Functions -----------------------------


def plot_fMRI_slice(bold_data, orientation="axial"):
    """
    Plots a selected slice from the 4D BOLD fMRI data based on the specified orientation.

    Parameters:
        bold_data (numpy.ndarray): The 4D BOLD data with shape (X, Y, Z, T).
        orientation (str): The orientation of the slice to plot ('axial', 'coronal', 'sagittal').

    Raises:
        ValueError: If an invalid orientation is specified.
    """
    # Determine the slice based on the orientation
    if orientation == "axial":
        slice_data = bold_data[:, :, bold_data.shape[2] // 2, bold_data.shape[3] // 2]
        title = "Axial Slice (Middle Volume)"
        xlabel = "Left \u2194 Right"
        ylabel = "Anterior \u2194 Posterior"
    elif orientation == "coronal":
        slice_data = bold_data[:, bold_data.shape[1] // 2, :, bold_data.shape[3] // 2]
        title = "Coronal Slice (Middle Volume)"
        xlabel = "Left \u2194 Right"
        ylabel = "Superior \u2194 Inferior"
    elif orientation == "sagittal":
        slice_data = bold_data[bold_data.shape[0] // 2, :, :, bold_data.shape[3] // 2]
        title = "Sagittal Slice (Middle Volume)"
        xlabel = "Anterior \u2194 Posterior"
        ylabel = "Superior \u2194 Inferior"
    else:
        raise ValueError("Invalid orientation specified. Choose 'axial', 'coronal', or 'sagittal'.")

    # Plot the selected slice
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_data.T, cmap="gray", origin="lower")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.show()


def plot_voxel_time_series(bold_data, voxel_coords):
    """
    Extracts and plots time-course data for the selected voxels.

    Parameters:
        bold_data (numpy.ndarray): The 4D BOLD data with shape (X, Y, Z, T).
        voxel_coords (list of tuples): List of voxel coordinates (x, y, z).

    Returns:
        dict: A dictionary containing time-series data for each voxel.
    """
    time_series = {}

    # Set up the plot
    plt.figure(figsize=(12, 6))

    for x, y, z in voxel_coords:
        # Extract the time-course
        time_series[(x, y, z)] = bold_data[x, y, z, :]

        # Plot the time-course
        plt.plot(time_series[(x, y, z)], label=f'Voxel ({x}, {y}, {z})')

    # Add plot details
    plt.title("Time-Course Data for Selected Voxels")
    plt.xlabel("Time (Volumes)")
    plt.ylabel("Signal Intensity")
    plt.legend()
    plt.show()

    # Print mean and standard deviation of the signal for each voxel
    for coords, signal in time_series.items():
        print(f"Voxel {coords}: Mean={np.mean(signal):.2f}, Std={np.std(signal):.2f}")

    return time_series

def plot_conditions_over_time(labels):
    """
    Maps conditions to numeric values and plots them over time.

    Parameters:
        labels (pandas.DataFrame): A DataFrame containing a 'Condition' column.

    Returns:
        dict: A mapping of conditions to numeric values.
    """
    # Map conditions to numeric values for visualization
    condition_map = {condition: idx for idx, condition in enumerate(labels["Condition"].unique())}
    condition_numeric = labels["Condition"].map(condition_map)

    # Plot conditions over time
    plt.figure(figsize=(12, 6))
    plt.plot(condition_numeric, label="Stimulus Condition", linestyle="None", marker="o", markersize=2)
    plt.title("Stimuli Over Time")
    plt.xlabel("Time (Volumes)")
    plt.ylabel("Condition (Numeric)")
    plt.yticks(list(condition_map.values()), list(condition_map.keys()))
    plt.grid()
    plt.show()

    # Plot distribution of conditions across runs
    plt.figure(figsize=(12, 6))
    sns.histplot(data=labels, x="Run", hue="Condition", multiple="stack")
    plt.title("Distribution of Conditions Across Runs")
    plt.xlabel("Run")
    plt.ylabel("Count")
    plt.show()

    return condition_map


def create_design_matrix(labels):
    """
    Creates a design matrix from a labels DataFrame.

    Parameters:
        labels (pandas.DataFrame): A DataFrame containing 'Condition' and 'Run' columns.

    Returns:
        pandas.DataFrame: The design matrix with conditions as columns.
    """
    # Get unique conditions from labels
    unique_conditions = labels["Condition"].unique()

    # Initialize the design matrix
    design_matrix = pd.DataFrame(0, index=np.arange(len(labels)), columns=unique_conditions)

    # Fill the design matrix
    for condition in unique_conditions:
        design_matrix[condition] = (labels["Condition"] == condition).astype(int)

    # Drop the 'rest' condition
    if "rest" in design_matrix.columns:
        design_matrix = design_matrix.drop(columns=["rest"])
        #print("'rest' condition dropped from the design matrix.")

    return design_matrix

def add_run_intercepts(design_matrix, labels):
    """
    Adds intercept columns for each run to the design matrix.

    Parameters:
        design_matrix (pd.DataFrame): Original design matrix (time points × conditions).
        labels (pd.DataFrame): DataFrame with 'Condition' and 'Run' columns.

    Returns:
        pd.DataFrame: Design matrix with added intercept columns.
    """
    design_matrix_with_intercepts = design_matrix.copy()
    unique_runs = labels["Run"].unique()

    # Add intercept columns
    for run in unique_runs:
        run_column_name = f"Run_{run}"
        design_matrix_with_intercepts[run_column_name] = (labels["Run"] == run).astype(int)

    return design_matrix_with_intercepts

def convolve_conditions(design_matrix, hrf_sampled):
    """
    Convolves only the condition columns of the design matrix with the HRF.

    Parameters:
        design_matrix (pd.DataFrame): Design matrix with conditions and intercepts.
        hrf_sampled (numpy.ndarray): The sampled HRF.

    Returns:
        pd.DataFrame: Design matrix with convolved conditions and unmodified intercepts.
    """
    # Identify condition columns (non-intercept)
    condition_columns = [col for col in design_matrix.columns if not col.startswith("Run")]
    intercept_columns = [col for col in design_matrix.columns if col.startswith("Run")]

    # Initialize the convolved matrix
    convolved_matrix = pd.DataFrame(index=design_matrix.index)

    # Convolve conditions
    for condition in condition_columns:
        convolved_signal = np.convolve(design_matrix[condition], hrf_sampled, mode="full")[:len(design_matrix)]
        convolved_matrix[condition] = convolved_signal

    # Add intercept columns without modification
    for intercept in intercept_columns:
        convolved_matrix[intercept] = design_matrix[intercept]

    return convolved_matrix


def fit_glm_and_generate_beta_maps(bold_data, X):
    """
    Fits a General Linear Model (GLM) voxel-by-voxel and generates beta coefficient maps.

    Parameters:
        bold_data (numpy.ndarray): The 4D BOLD fMRI data with shape (X, Y, Z, Time).
        X (pandas.DataFrame): The convolved design matrix.

    Returns:
        numpy.ndarray: Beta coefficient maps with shape (X, Y, Z, N_conditions).
    """
    # Reshape the fMRI data for voxel-wise analysis
    Y = bold_data.reshape(-1, bold_data.shape[-1]).T  # Shape: (Time, Voxels)
    # print(f"fMRI data reshaped to: {Y.shape}")

    # Ensure the design matrix is aligned with the fMRI data
    assert Y.shape[0] == X.shape[0], "Mismatch in time points!"

    # Solve the GLM
    X_pinv = np.linalg.pinv(X)  # Use pseudo-inverse for stability
    beta_matrix = X_pinv @ Y

    # Compute residuals (optional, for model evaluation)
    residuals = Y - (X @ beta_matrix)  # Shape: (Time, N_voxels)

    # Reshape beta coefficients back to brain space
    beta_maps = beta_matrix.T.reshape(bold_data.shape[:-1] + (X.shape[1],))  # Shape: (X, Y, Z, N_conditions)

    return residuals, beta_maps


def visualize_beta_map(beta_maps, convolved_matrix, condition_name, slice_idx=None):
    """
    Visualizes the beta map for a specified condition with optional slice selection and thresholding.

    Parameters:
        beta_maps (numpy.ndarray): Beta coefficient maps with shape (X, Y, Z, N_conditions).
        convolved_matrix (pandas.DataFrame): The convolved design matrix.
        condition_name (str): The name of the condition to visualize.
        slice_idx (int, optional): The z-dimension slice index to visualize. Defaults to the middle slice.

    Raises:
        ValueError: If the specified condition is not found in the design matrix.
    """
    # Find the column index for the specified condition
    if condition_name not in convolved_matrix.columns:
        raise ValueError(f"Condition '{condition_name}' not found in the design matrix.")

    condition_idx = list(convolved_matrix.columns).index(condition_name)

    # Default to middle slice if slice_idx is not provided
    if slice_idx is None:
        slice_idx = beta_maps.shape[2] // 2

    # Extract the beta map for the specified condition and slice
    beta_map_slice = beta_maps[:, :, slice_idx, condition_idx]

    # Apply thresholding
    threshold = 0.1
    beta_map_slice = np.where(np.abs(beta_map_slice) < threshold, 0, beta_map_slice)

    # Visualize the beta map
    plt.figure(figsize=(8, 8))
    plt.imshow(beta_map_slice.T, cmap="coolwarm", origin="lower")
    plt.title(f"Beta Map for Condition: {condition_name} (Slice {slice_idx}, Threshold {threshold})")
    plt.colorbar(label="Beta Value")
    plt.xlabel("X-axis (Left ↔ Right)")
    plt.ylabel("Y-axis (Anterior ↔ Posterior)")
    plt.show()


def visualize_beta_map_grid(beta_maps, convolved_matrix, condition_name, slice_range=None, slices_per_row=7):
    """
    Visualizes beta maps for a specified condition in a grid layout.

    Parameters:
        beta_maps (numpy.ndarray): Beta coefficient maps with shape (X, Y, Z, N_conditions).
        convolved_matrix (pandas.DataFrame): The convolved design matrix.
        condition_name (str): The name of the condition to visualize.
        slice_range (list, optional): A list of z-dimension slice indices to visualize. Defaults to all slices.
        slices_per_row (int): Number of slices to display per row. Defaults to 7.

    Raises:
        ValueError: If the specified condition is not found in the design matrix.
    """

    # Validate the condition
    if condition_name not in convolved_matrix.columns:
        raise ValueError(f"Condition '{condition_name}' not found in the design matrix.")

    condition_idx = convolved_matrix.columns.get_loc(condition_name)

    # Set the slice range to all slices if not specified
    if slice_range is None:
        slice_range = range(beta_maps.shape[2])

    # Determine grid size
    n_slices = len(slice_range)
    n_rows = int(np.ceil(n_slices / slices_per_row))

    # Initialize the plot
    fig, axes = plt.subplots(n_rows, slices_per_row,
                             figsize=(slices_per_row * 2, n_rows * 2),
                             constrained_layout=True)

    axes = axes.flatten()  # Flatten for easy indexing

    # Plot each slice
    for i, slice_idx in enumerate(slice_range):
        if i >= len(axes):
            break

        # Extract the beta map for the current slice and condition
        beta_map_slice = beta_maps[:, :, slice_idx, condition_idx]

        # Display the beta map
        im = axes[i].imshow(beta_map_slice, cmap="coolwarm", origin="lower", vmin=-10, vmax=10)
        axes[i].axis("off")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Add a single shared colorbar
    fig.colorbar(im, ax=axes[:n_slices], orientation="vertical", fraction=0.02, pad=0.04, label="Beta Value")

    # Add a title
    fig.suptitle(f"Beta Maps for Condition: {condition_name}", fontsize=14)
    plt.show()


def compute_t_values(beta_maps, design_matrix, residuals, df, mask=None, variance_threshold=1e-12, epsilon=1e-10):
    """
    Computes t-values for beta maps and residuals.

    Parameters
    ----------
    beta_maps : np.ndarray
        4D array of beta coefficients, shape (X, Y, Z, N_conditions).
    design_matrix : pd.DataFrame
        Design matrix, shape (time_points, N_conditions).
    residuals : np.ndarray
        Residuals, shape (time_points, N_voxels).
    df : int
        Degrees of freedom (time_points - rank(X)).
    mask : np.ndarray, optional
        Boolean array (X, Y, Z). Restricts computations if provided.
    variance_threshold : float, optional
        Minimum residual variance threshold.
    epsilon : float, optional
        Small value to avoid division by zero.

    Returns
    -------
    t_maps : np.ndarray
        4D array of t-values, shape (X, Y, Z, N_conditions).
    """
    X = design_matrix.values
    X_pinv = np.linalg.pinv(X)
    diag_cov = np.diag(X_pinv @ X_pinv.T).astype(float)

    # Handle zero values in diag_cov
    diag_cov = np.maximum(diag_cov, epsilon)
    # diag_cov = np.diag(np.linalg.pinv(design_matrix.values.T @ design_matrix.values))

    n_time_points, n_regressors = X.shape
    X_size, Y_size, Z_size, n_conditions = beta_maps.shape
    assert n_regressors == n_conditions, "Mismatch between design matrix and beta maps."

    beta_reshaped = beta_maps.reshape(-1, n_conditions)
    n_voxels = X_size * Y_size * Z_size
    assert residuals.shape == (n_time_points, n_voxels), "Residuals shape mismatch."

    if mask is not None:
        assert mask.shape == (X_size, Y_size, Z_size), "Invalid mask shape."
        mask_indices = np.where(mask.flatten())
        beta_reshaped = beta_reshaped[mask_indices]
        residuals = residuals[:, mask_indices[0]]
    else:
        mask_indices = None

    residual_var = np.sum(residuals**2, axis=0) / df
    residual_var = np.maximum(residual_var, variance_threshold)

    # Ensure residual_var is a NumPy array
    residual_var = np.asarray(residual_var)

    # Compute standard errors and enforce epsilon for safety
    SE = np.sqrt(residual_var[:, np.newaxis] * diag_cov)
    SE = np.maximum(SE, epsilon)

    # Compute t-values
    t_values = beta_reshaped / SE

    # Reconstruct t-maps
    t_maps = np.zeros((X_size, Y_size, Z_size, n_conditions), dtype=np.float32)
    if mask is not None:
        t_maps_flat = t_maps.reshape(-1, n_conditions)
        t_maps_flat[mask_indices] = t_values
        t_maps = t_maps_flat.reshape(X_size, Y_size, Z_size, n_conditions)
    else:
        t_maps = t_values.reshape(X_size, Y_size, Z_size, n_conditions)

    return t_maps


def visualize_t_map(t_maps, design_matrix, condition_name,
                            slice_idx=None, cmap="bwr", vmin=-10, vmax=10):
    """
    Visualizes the t-map for a specified condition. If slice_idx is None,
    use the middle slice.

    Parameters
    ----------
    t_maps : np.ndarray
        4D array of t-values, shape (X, Y, Z, N_conditions).
    design_matrix : pd.DataFrame
        The design matrix (to find condition_name index).
    condition_name : str
        The condition to visualize (must be a column in the design matrix).
    slice_idx : int or None
        Which z-slice to display. Defaults to the middle.
    """
    if condition_name not in design_matrix.columns:
        raise ValueError(f"Condition '{condition_name}' not found in the design matrix columns.")

    condition_idx = list(design_matrix.columns).index(condition_name)
    X_size, Y_size, Z_size, n_conditions = t_maps.shape

    if slice_idx is None:
        slice_idx = Z_size // 2  # middle slice

    # Extract 2D slice for the chosen condition
    t_slice = t_maps[:, :, slice_idx, condition_idx]

    plt.figure(figsize=(6, 5))
    plt.imshow(t_slice, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(label="t-value")
    plt.title(f"T-Map for Condition: {condition_name}\nSlice {slice_idx}")
    plt.axis("off")
    plt.show()


def overlay_t_map_on_fmri(fmri_data, t_map, slice_idx, condition_name, alpha=0.5, cmap="coolwarm"):
    """
    Overlays a t-map on top of fMRI data for a specific slice with improved clarity.

    Parameters:
        fmri_data (numpy.ndarray): The 4D fMRI data (X, Y, Z, Time).
        t_map (numpy.ndarray): The full-brain t-map (X, Y, Z).
        slice_idx (int): The index of the z-dimension slice to visualize.
        condition_name (str): The name of the condition for the t-map.
        alpha (float, optional): Opacity for the t-map overlay. Default is 0.5.
        cmap (str, optional): Colormap for the t-map. Default is "coolwarm".
    """
    if slice_idx < 0 or slice_idx >= fmri_data.shape[2]:
        raise ValueError(f"Invalid slice index {slice_idx}. Must be between 0 and {fmri_data.shape[2] - 1}.")

    # Extract the anatomical slice from the fMRI data (mean across time for visualization)
    fmri_slice = fmri_data[:, slice_idx, :, :].mean(axis=-1)

    # Normalize and apply gamma correction
    fmri_slice = (fmri_slice - fmri_slice.min()) / (fmri_slice.max() - fmri_slice.min())
    gamma = 0.5
    fmri_slice = np.power(fmri_slice, gamma)

    # Extract the corresponding t-map slice and apply thresholding
    t_map_slice = t_map[:, :, slice_idx]
    threshold = 2.0
    t_map_slice = np.where(np.abs(t_map_slice) < threshold, 0, t_map_slice)

    # Plot the overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(fmri_slice.T, cmap="gray", origin="lower")
    plt.imshow(t_map_slice.T, cmap=cmap, alpha=alpha, origin="lower")
    plt.colorbar(label="t-value")
    plt.title(f"Overlay of T-Map and fMRI Data (Slice {slice_idx})\nCondition: {condition_name}")
    plt.xlabel("X-axis (Left ↔ Right)")
    plt.ylabel("Y-axis (Anterior ↔ Posterior)")
    plt.show()

def apply_bonferroni_correction(t_maps, df, mask_bool, alpha=0.05):
    """
    Applies Bonferroni correction to T-maps.

    Parameters:
        t_maps (numpy.ndarray): Full-brain T-maps with shape (X, Y, Z, N_conditions).
        df (int): Degrees of freedom (time points - rank(X)).
        mask_bool (numpy.ndarray): Boolean mask for the region of interest (ROI).
        alpha (float, optional): Desired family-wise error rate (default: 0.05).

    Returns:
        numpy.ndarray: Corrected T-maps with non-significant values set to 0.
    """
    # Number of tests (voxels in the mask)
    n_tests = np.sum(mask_bool)
    #print(f"Number of tests (voxels): {n_tests}")

    # Adjusted p-value threshold using Bonferroni correction
    adjusted_alpha = alpha / n_tests
    #print(f"Bonferroni-adjusted alpha: {adjusted_alpha}")

    # Convert the adjusted alpha to a T-value threshold
    t_threshold = t.ppf(1 - adjusted_alpha / 2, df)  # Two-tailed
    #print(f"Bonferroni T-value threshold: {t_threshold}")

    # Mask the T-maps with the Bonferroni threshold
    corrected_t_maps = np.zeros_like(t_maps)
    significant_mask = np.abs(t_maps) > t_threshold  # Significant voxels
    corrected_t_maps[significant_mask] = t_maps[significant_mask]

    return corrected_t_maps


def create_convolved_matrix_with_trends(design_matrix, max_degree=2):
    """
    Creates a convolved design matrix by convolving each condition with the HRF and adds constant terms
    and polynomial trends for high-pass filtering.

    Parameters:
        design_matrix (pandas.DataFrame): The original design matrix with conditions as columns.
        hrf_sampled (numpy.ndarray): The sampled hemodynamic response function (HRF).
        max_degree (int, optional): The maximum degree of polynomial trends to include. Default is 2 (linear and quadratic).

    Returns:
        pandas.DataFrame: The enhanced convolved design matrix.
    """
    # Add constant term (intercept)
    #convolved_matrix['constant'] = 1

    # Add polynomial trends
    time = np.arange(len(design_matrix))
    for degree in range(1, max_degree + 1):
        trend_name = f"poly_{degree}"
        design_matrix[trend_name] = (time**degree - (time**degree).mean()) / (time**degree).std()

    return design_matrix



# ------------------------------------------------------------------------------------


# ----------------------------- Project Work 2 Functions -----------------------------

def compute_contrast_map(beta_maps, design_matrix, contrast_vector, df, residual_variance, mask=None, variance_threshold=1e-6):
    """
    Computes a contrast map and voxelwise t-values.

    Parameters
    ----------
    beta_maps : np.ndarray
        4D array of beta coefficients, shape (X, Y, Z, N_conditions).
    design_matrix : pd.DataFrame
        Design matrix, shape (time_points, N_conditions).
    contrast_vector : array-like
        Contrast vector, length N_conditions (e.g., [1, -1, 0, ...]).
    df : int
        Degrees of freedom (time_points - rank(X)).
    residual_variance : np.ndarray
        Voxelwise residual variance, shape (X, Y, Z) or (n_mask_voxels,).
    mask : np.ndarray, optional
        Boolean array, shape (X, Y, Z). Restricts computations if provided.
    variance_threshold : float, optional
        Minimum variance threshold to avoid division by zero.

    Returns
    -------
    t_map_full : np.ndarray
        3D array of t-values, shape (X, Y, Z).
    """
    X = design_matrix.values
    X_size, Y_size, Z_size, n_conditions = beta_maps.shape
    assert n_conditions == X.shape[1], "Mismatch between beta_maps and design_matrix."
    contrast_vector = np.array(contrast_vector)
    assert len(contrast_vector) == n_conditions, "Mismatch between contrast_vector and design_matrix."

    cov_contrast = contrast_vector.T @ np.linalg.pinv(X) @ np.linalg.pinv(X).T @ contrast_vector
    contrast_betas = np.sum(beta_maps * contrast_vector, axis=3)

    if mask is not None:
        mask_flat = mask.flatten()
        residual_variance = residual_variance[mask] if residual_variance.ndim == 3 else residual_variance
        residual_variance = np.maximum(residual_variance, variance_threshold)

        SE = np.sqrt(residual_variance * cov_contrast)
        t_values = contrast_betas[mask] / SE

        t_map_full = np.full((X_size, Y_size, Z_size), np.nan, dtype=np.float32)
        t_map_full[mask] = t_values
    else:
        residual_variance = np.maximum(residual_variance, variance_threshold)
        SE = np.sqrt(residual_variance * cov_contrast)
        t_map_full = (contrast_betas / SE).astype(np.float32)

    return t_map_full

def load_subject_data(bold_path, labels_path):
    """
    Load fMRI BOLD data and labels for a subject.

    Parameters:
        bold_path (str): Path to the subject's BOLD data (NIfTI file).
        labels_path (str): Path to the subject's labels file (CSV).

    Returns:
        bold_data (numpy.ndarray): 4D BOLD data array.
        labels (pandas.DataFrame): Labels DataFrame with 'Condition' and 'Run'.
    """
    # Load the fMRI BOLD data
    bold_img = nib.load(bold_path)
    bold_data = bold_img.get_fdata()

    # Load the labels
    labels = pd.read_csv(labels_path, sep=" ", header=0, names=["Condition", "Run"])

    return bold_data, labels

def calculate_roi_percent_signal_change(beta_maps, roi_masks):
    """
    Calculate percent signal change for each ROI.

    Parameters:
        beta_maps (numpy.ndarray): Beta coefficient maps.
        roi_masks (dict): Dictionary of ROI masks.

    Returns:
        roi_percent_signal_change (dict): Percent signal change for each ROI.
        roi_mean_percent_change (dict): Mean percent signal change across voxels in each ROI.
        roi_sem (dict): SEM for percent signal change in each ROI.
    """
    roi_percent_signal_change = {}
    roi_mean_percent_change = {}
    roi_sem = {}

    constant_term_indices = list(range(-12, 0))  # Last 12 columns are the constant terms

    for roi_name, mask in roi_masks.items():
        roi_betas = beta_maps[mask]  # Extract beta values for this ROI
        #print(f"Shape of beta maps for ROI '{roi_name}': {roi_betas.shape}")  # Debugging

        # Skip ROIs with no voxels
        if roi_betas.shape[0] == 0:
            print(f"Skipping ROI '{roi_name}' due to no voxels.")
            continue

        # Exclude constant terms for conditions
        condition_betas = roi_betas[:, :-12]
        constant_terms = roi_betas[:, constant_term_indices]
        mean_constant_term = np.mean(constant_terms, axis=1)

        # Calculate percent signal change
        percent_change = (condition_betas / mean_constant_term[:, np.newaxis]) * 100
        roi_percent_signal_change[roi_name] = percent_change  # Shape: (n_voxels, n_conditions)

        # Aggregate across voxels
        roi_mean_percent_change[roi_name] = np.mean(percent_change, axis=0)  # Mean: (n_conditions,)
        roi_sem[roi_name] = (
            np.std(percent_change, axis=0) / np.sqrt(percent_change.shape[0])  # SEM: (n_conditions,)
            if percent_change.shape[0] > 1
            else np.zeros(percent_change.shape[1])  # Handle single-voxel ROIs
        )

        # Debugging: Print shapes
        #print(f"Shape of percent change for ROI '{roi_name}': {percent_change.shape}")
        #print(f"Mean percent signal change for ROI '{roi_name}': {roi_mean_percent_change[roi_name].shape}")
        #print(f"SEM for ROI '{roi_name}': {roi_sem[roi_name].shape}")

    return roi_percent_signal_change, roi_mean_percent_change, roi_sem

def plot_roi_mean_percent_change(roi_mean_percent_change, roi_sem, unique_conditions, colors):
    """
    Plot bar graphs for mean percent signal changes.

    Parameters:
        roi_mean_percent_change (dict): Mean percent signal change for each ROI.
        roi_sem (dict): SEM for each ROI.
        unique_conditions (list): List of condition names.
        colors (list): List of colors for the bars.
    """
    n_rois = len(roi_mean_percent_change)
    n_cols = 2
    n_rows = int(np.ceil(n_rois / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 4))
    axes = axes.flatten()

    # Concatenate all values to calculate global y-axis limits
    all_mean_values = np.concatenate([values for values in roi_mean_percent_change.values()])
    all_sem_values = np.concatenate([values for values in roi_sem.values()])
    y_min = -1.5
    y_max = (all_mean_values + all_sem_values).max() * 1.1

    for idx, (roi_name, mean_values) in enumerate(roi_mean_percent_change.items()):
        ax = axes[idx]
        sem_values = roi_sem[roi_name]

        # Plot bar graph
        ax.bar(
            range(len(mean_values)),
            mean_values,
            yerr=sem_values,
            color=colors[:len(mean_values)],
            alpha=0.7,
            capsize=5,
            edgecolor="black"
        )
        ax.set_title(f"{roi_name}", fontsize=12)
        ax.set_xlabel("Conditions", fontsize=10)
        ax.set_ylabel("Percent Signal Change (%)", fontsize=10)
        ax.set_xticks(np.arange(len(unique_conditions)))
        ax.set_xticklabels(unique_conditions, rotation=45, fontsize=8)
        ax.set_ylim(y_min, y_max)

    for idx in range(len(roi_mean_percent_change), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()

def analyze_subject(subject_id, bold_path, labels_path, hrf_path, unique_conditions, colors):
    """
    Perform subject-level analysis: GLM fitting, ROI signal extraction, and visualization.

    Parameters:
        subject_id (str): Subject ID.
        bold_path (str): Path to the subject's BOLD data (NIfTI file).
        labels_path (str): Path to the subject's labels file (CSV).
        hrf_path (str): Path to the HRF data (MAT file).
        unique_conditions (list): List of condition names.
        colors (list): List of colors for visualization.

    Returns:
        roi_mean_percent_change (dict): Mean percent signal change for each ROI.
    """
    # Step 1: Load Subject Data
    print(f"Loading subject {subject_id} data...")
    bold_data, labels = load_subject_data(bold_path, labels_path)

    # Step 2: Load ROI Masks
    print(f"Loading ROI masks for subject {subject_id}...")
    roi_masks = load_subject_roi_masks(subject_id)

    # Step 3: Create Design Matrix
    print("Creating design matrix...")
    design_matrix = create_design_matrix(labels)

    # Step 4: Modify Convolved Matrix
    print("Modifying design matrix...")
    design_matrix_with_intercepts = add_run_intercepts(design_matrix, labels)

    # Step 5: Load HRF and Create Convolved Matrix
    print("Loading HRF and convolving design matrix...")
    hrf_data = loadmat(hrf_path)
    hrf_sampled = hrf_data.get("hrf_sampled", None).flatten()  # Use the downsampled HRF
    convolved_matrix = convolve_conditions(design_matrix_with_intercepts, hrf_sampled)
    assert bold_data.shape[-1] == convolved_matrix.shape[0], "Mismatch between BOLD volumes and design matrix rows."

    # Step 6: Fit GLM
    print("Fitting GLM...")
    residuals, beta_maps = fit_glm_and_generate_beta_maps(bold_data, convolved_matrix.values)

    # Step 7: ROI Analysis
    print("Extracting ROI signal changes...")
    roi_percent_signal_change, roi_mean_percent_change, roi_sem = calculate_roi_percent_signal_change(beta_maps, roi_masks)

    # Step 8: Visualization
    #print("Visualizing ROI results...")
    #plot_roi_mean_percent_change(roi_mean_percent_change, roi_sem, unique_conditions, colors)

    print(f"Subject {subject_id} analysis complete.")
    return roi_mean_percent_change

def load_subject_roi_masks(subject_id):
    """
    Load ROI masks for a specific subject using global base paths.

    Parameters:
        subject_id (str): Subject ID.

    Returns:
        dict: Dictionary of ROI masks for the subject.
    """
    # Generate full paths for this subject
    vt_mask_path = base_vt_mask_path.format(subject_id)
    face_mask_path = base_face_mask_path.format(subject_id)
    house_mask_path = base_house_mask_path.format(subject_id)

    # Load the masks
    vt_mask = nib.load(vt_mask_path).get_fdata() > 0
    face_mask = nib.load(face_mask_path).get_fdata() > 0
    house_mask = nib.load(house_mask_path).get_fdata() > 0

    # Generate additional masks if needed
    brain_mask = vt_mask | face_mask | house_mask
    random_roi_inside = np.random.choice([False, True], size=vt_mask.shape, p=[0.995, 0.005]) & brain_mask
    random_roi_outside = np.random.choice([False, True], size=vt_mask.shape, p=[0.999, 0.001]) & ~brain_mask

    return {
        "Ventral Temporal": vt_mask,
        "Face": face_mask,
        "House": house_mask,
        "Random Inside": random_roi_inside,
        "Random Outside": random_roi_outside,
    }


def analyze_multiple_subjects(subject_ids, base_bold_path, base_labels_path, hrf_path, unique_conditions, colors):
    """
    Perform analysis across multiple subjects and average results across them.

    Parameters:
        subject_ids (list): List of subject IDs (e.g., ["1", "2", "3", "4"]).
        base_bold_path (str): Base path for BOLD files (e.g., "subj{}/bold.nii.gz").
        base_labels_path (str): Base path for labels files (e.g., "subj{}/labels.txt").
        hrf_path (str): Path to the HRF file (shared across subjects).
        roi_masks (dict): Dictionary of ROI masks (same across subjects).
        unique_conditions (list): List of condition names.
        colors (list): List of colors for visualization.

    Returns:
        group_mean_percent_change (dict): Mean percent signal change across subjects for each ROI.
        group_sem (dict): Standard error of the mean (SEM) across subjects for each ROI.
    """
    # Store subject-level results
    all_subjects_percent_change = {roi_name: [] for roi_name in roi_masks.keys()}

    # Iterate over all subjects
    for subject_id in subject_ids:
        print(f"Analyzing subject {subject_id}...")

        # Generate subject-specific paths
        bold_path = base_bold_path.format(subject_id)
        labels_path = base_labels_path.format(subject_id)

        # Perform single-subject analysis
        roi_mean_percent_change = analyze_subject(
            subject_id, bold_path, labels_path, hrf_path, unique_conditions, colors
        )

        # Collect results for group analysis
        for roi_name, mean_percent_change in roi_mean_percent_change.items():
            all_subjects_percent_change[roi_name].append(mean_percent_change)

    # Calculate group-level averages
    group_mean_percent_change = {}
    group_sem = {}

    for roi_name, subject_values in all_subjects_percent_change.items():
        subject_values = np.array(subject_values)  # Convert to numpy array for easier manipulation
        group_mean_percent_change[roi_name] = np.mean(subject_values, axis=0)  # Mean across subjects
        group_sem[roi_name] = np.std(subject_values, axis=0) / np.sqrt(len(subject_values))  # SEM

    # Visualize group results
    print("Visualizing group results...")
    plot_roi_mean_percent_change(group_mean_percent_change, group_sem, unique_conditions, colors)

    print("Group analysis complete.")
    return group_mean_percent_change, group_sem



# ------------------------------------------------------------------------------------


# ----------------------------- Project Work 3 Functions -----------------------------

def split_data_by_runs(labels, convolved_matrix, bold_data):
    """
    Splits the data into even and odd runs and ensures proper intercept handling.

    Parameters:
        labels (pd.DataFrame): DataFrame containing 'Condition' and 'Run'.
        convolved_matrix (pd.DataFrame): Full design matrix (time points × conditions + intercepts).
        bold_data (np.ndarray): BOLD fMRI data with shape (X, Y, Z, Time).

    Returns:
        dict: Dictionary with keys 'even' and 'odd', each containing:
              - "design_matrix": Split design matrix for even/odd runs (time points/2 × 14).
              - "bold_data": Corresponding split BOLD data.
    """
    # Get indices for even and odd runs
    even_indices = labels.index[labels["Run"] % 2 == 0]
    odd_indices = labels.index[labels["Run"] % 2 == 1]

    # Extract condition columns and appropriate intercepts for even runs
    even_conditions = [col for col in convolved_matrix.columns if not col.startswith("Run_")]
    even_intercepts = [f"Run_{run}" for run in labels["Run"].unique() if run % 2 == 0]
    even_columns = even_conditions + even_intercepts

    # Extract condition columns and appropriate intercepts for odd runs
    odd_conditions = [col for col in convolved_matrix.columns if not col.startswith("Run_")]
    odd_intercepts = [f"Run_{run}" for run in labels["Run"].unique() if run % 2 == 1]
    odd_columns = odd_conditions + odd_intercepts

    # Create split design matrices
    even_design_matrix = convolved_matrix.loc[even_indices, even_columns]
    odd_design_matrix = convolved_matrix.loc[odd_indices, odd_columns]

    # Split BOLD data
    even_bold_data = bold_data[..., even_indices]
    odd_bold_data = bold_data[..., odd_indices]

    # Return results
    return {
        "even": {
            "design_matrix": even_design_matrix,
            "bold_data": even_bold_data,
        },
        "odd": {
            "design_matrix": odd_design_matrix,
            "bold_data": odd_bold_data,
        },
    }


def visualize_design_matrix(design_matrix, title):
    """
    Visualizes the design matrix with enhancements for better column clarity.

    Parameters:
        design_matrix (pd.DataFrame): The design matrix to visualize.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(14, 8))  # Increase figure size for clarity
    plt.imshow(design_matrix.values, aspect="auto", cmap="gray", interpolation="none")
    plt.colorbar(label="Regressor Value")

    # Add vertical gridlines to separate columns
    num_columns = design_matrix.shape[1]
    plt.xticks(ticks=range(num_columns), labels=design_matrix.columns, rotation=90)
    plt.grid(axis="x", color="white", linestyle="--", linewidth=0.5)

    plt.title(title)
    plt.xlabel("Regressors")
    plt.ylabel("Time Points")
    plt.tight_layout()
    plt.show()


# Step 2: Fit GLMs and compute T-maps
def fit_glm_and_compute_tmaps(split_data, df):
    """
    Fits GLMs and computes T-maps for the given split data.

    Parameters:
        split_data (dict): Dictionary containing 'even' and 'odd' subsets.
        df (int): Degrees of freedom for GLM.

    Returns:
        dict: Dictionary with beta maps, residuals, and t-maps for even and odd runs.
    """
    results = {}
    for run_type in ["even", "odd"]:
        design_matrix = split_data[run_type]["design_matrix"]
        bold_data = split_data[run_type]["bold_data"]

        # Fit the GLM
        residuals, beta_maps = fit_glm_and_generate_beta_maps(bold_data, design_matrix)

        # Compute T-maps
        t_maps = compute_t_values(
            beta_maps=beta_maps,
            design_matrix=design_matrix,
            residuals=residuals,
            df=df
        )

        results[run_type] = {
            "beta_maps": beta_maps,
            "residuals": residuals,
            "t_maps": t_maps,
        }

    return results


# Extract t-maps for a condition
def extract_condition_t_maps(glm_results, design_matrix, condition_name):
    """
    Extracts the t-maps corresponding to a specific condition.

    Parameters:
        glm_results (dict): Dictionary with t-maps for even and odd runs.
        design_matrix (pd.DataFrame): The original design matrix.
        condition_name (str): The condition of interest.

    Returns:
        dict: T-maps for the given condition (keys: 'even', 'odd').
    """
    if condition_name not in design_matrix.columns:
        raise ValueError(f"Condition '{condition_name}' not found in the design matrix.")

    condition_idx = list(design_matrix.columns).index(condition_name)

    return {
        run_type: glm_results[run_type]["t_maps"][..., condition_idx]
        for run_type in ["even", "odd"]
    }

# Apply ROI mask to t-maps
def apply_mask_to_t_maps(t_maps, mask):
    """
    Applies an ROI mask to the t-maps.
    """
    return t_maps[mask]

# Compute and verify within- and between-condition correlations
def compute_and_verify_correlations(glm_results, design_matrix, mask):
    """
    Computes within- and between-condition correlations for all conditions
    and checks if within-correlation is higher than all between-correlations.

    Parameters:
        glm_results (dict): GLM results with t-maps for even and odd runs.
        design_matrix (pd.DataFrame): Original design matrix.
        mask (np.ndarray): 3D boolean ROI mask.

    Returns:
        dict: Dictionary of within- and between-condition correlations for each condition,
              with verification results.
    """
    true_conditions = [col for col in design_matrix.columns if not col.startswith("Run")]

    results = {}
    for condition in true_conditions:
        t_maps_condition = extract_condition_t_maps(glm_results, design_matrix, condition)
        masked_even = apply_mask_to_t_maps(t_maps_condition["even"], mask)
        masked_odd = apply_mask_to_t_maps(t_maps_condition["odd"], mask)

        # Compute within-condition correlation
        within_correlation = np.corrcoef(masked_even.flatten(), masked_odd.flatten())[0, 1]

        # Compute between-condition correlations
        between_correlations = {
            other_condition: np.corrcoef(masked_even.flatten(),
                                         apply_mask_to_t_maps(
                                             extract_condition_t_maps(glm_results, design_matrix, other_condition)["odd"], mask).flatten()
                                         )[0, 1]
            for other_condition in true_conditions if other_condition != condition
        }

        is_higher = all(within_correlation > corr for corr in between_correlations.values())
        results[condition] = {
            "within": within_correlation,
            "between": between_correlations,
            "is_higher": is_higher,
        }

    return results

def plot_within_between_correlations(correlation_results):
    """
    Plots a separate bar chart for each condition in correlation_results.

    Parameters
    ----------
    correlation_results : dict
        Output from compute_and_verify_correlations, i.e.:
        {
          "face": {
            "within": float,
            "between": {"face vs. scissors": float, ...},
            "is_higher": bool
          },
          ...
        }
    """
    # Convert the dictionary format:
    # correlation_results[condition]["within"] -> float
    # correlation_results[condition]["between"] -> {other_condition: correlation, ...}
    # correlation_results[condition]["is_higher"] -> bool

    conditions = list(correlation_results.keys())
    n_conditions = len(conditions)

    # Create one subplot per condition
    fig, axes = plt.subplots(n_conditions, 1,
                             figsize=(8, 3 * n_conditions),
                             sharex=False)

    if n_conditions == 1:
        # If there's only one condition, make 'axes' a list for consistency
        axes = [axes]

    for i, condition in enumerate(conditions):
        ax = axes[i]
        corr_info = correlation_results[condition]

        # We'll plot 'Within' as one bar, and each "between" correlation as additional bars
        categories = ["Within"]
        corr_values = [corr_info["within"]]

        # Append the between-condition categories/values
        for other_cond, val in corr_info["between"].items():
            # The dictionary key is typically "face vs. scissors"
            # but you might prefer just "scissors" to keep the label short.
            if " vs. " in other_cond:
                # everything after ' vs. '
                short_label = other_cond.split(" vs. ")[-1]
            else:
                short_label = other_cond
            categories.append(short_label)
            corr_values.append(val)

        # Plot the bars
        x_positions = np.arange(len(categories))
        bars = ax.bar(x_positions, corr_values, color="skyblue")

        # Color the 'Within' bar specially if desired:
        if corr_info["is_higher"]:
            bars[0].set_color("tomato")  # e.g. red if it's higher than all between
        else:
            bars[0].set_color("gray")    # or gray if not

        # Labeling and cosmetics
        ax.set_title(f"Condition: {condition}")
        ax.set_ylabel("Correlation")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_ylim([-1, 1])  # typical correlation range

        # Optional: annotate each bar with its numeric value
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                np.sign(height) * (abs(height) + 0.02),  # a little above/below the bar
                f"{height:.2f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                color="black"
            )

    plt.tight_layout()
    plt.show()


def compute_group_correlations(subject_ids, base_bold_path, base_labels_path, hrf_path, base_vt_mask_path,
                               unique_conditions):
    """
    Compute within- and between-condition correlations for all subjects and average the results.

    Parameters:
        subject_ids (list): List of subject IDs.
        base_bold_path (str): Template path for subject-specific BOLD files.
        base_labels_path (str): Template path for subject-specific labels files.
        hrf_path (str): Path to the HRF file.
        base_vt_mask_path (str): Template path for subject-specific Ventral Temporal ROI masks.
        unique_conditions (list): List of unique conditions.

    Returns:
        dict: Group-level correlation results with mean and SEM for within and between conditions.
    """
    all_subject_results = {cond: {"within": [], "between": {}} for cond in unique_conditions}

    # Perform group-level analysis
    for subj_id in subject_ids:
        print(f"Processing subject {subj_id}...")

        # Generate paths for subject-specific data
        bold_path = base_bold_path.format(subj_id)
        labels_path = base_labels_path.format(subj_id)
        vt_mask_path = base_vt_mask_path.format(subj_id)  # Subject-specific VT mask path

        # Load subject-specific VT mask
        vt_mask = nib.load(vt_mask_path).get_fdata() > 0

        # Perform subject-level analysis
        bold_data, labels = load_subject_data(bold_path, labels_path)
        design_matrix = create_design_matrix(labels)
        design_matrix_with_intercepts = add_run_intercepts(design_matrix, labels)
        hrf_data = loadmat(hrf_path)["hrf_sampled"].flatten()
        convolved_matrix = convolve_conditions(design_matrix_with_intercepts, hrf_data)

        # Split data into even and odd runs and fit GLMs
        split_data = split_data_by_runs(labels, convolved_matrix, bold_data)
        df = split_data["even"]["design_matrix"].shape[0] - np.linalg.matrix_rank(
            split_data["even"]["design_matrix"].values)
        glm_results = fit_glm_and_compute_tmaps(split_data, df)

        # Compute correlations for this subject using the subject-specific VT mask
        subj_correlations = compute_and_verify_correlations(glm_results, convolved_matrix, vt_mask)

        # Aggregate correlations for group-level analysis
        for condition, results in subj_correlations.items():
            all_subject_results[condition]["within"].append(results["within"])
            for other_condition, corr in results["between"].items():
                if other_condition not in all_subject_results[condition]["between"]:
                    all_subject_results[condition]["between"][other_condition] = []
                all_subject_results[condition]["between"][other_condition].append(corr)

    # Compute group-level means and SEMs
    group_results = {}
    for condition, data in all_subject_results.items():
        group_results[condition] = {
            "within": {
                "mean": np.mean(data["within"]),
                "sem": np.std(data["within"]) / np.sqrt(len(data["within"]))
            },
            "between": {
                other_cond: {
                    "mean": np.mean(corrs),
                    "sem": np.std(corrs) / np.sqrt(len(corrs))
                }
                for other_cond, corrs in data["between"].items()
            }
        }

    return group_results


def plot_group_correlations(group_correlation_results):
    """
    Plot group-level within- and between-condition correlations.

    Parameters:
        group_correlation_results (dict): Group-level correlation results with mean and SEM.
    """
    conditions = list(group_correlation_results.keys())
    n_conditions = len(conditions)

    # Create one subplot per condition
    fig, axes = plt.subplots(n_conditions, 1, figsize=(8, 3 * n_conditions), sharex=False)

    if n_conditions == 1:
        axes = [axes]

    for i, condition in enumerate(conditions):
        ax = axes[i]
        group_data = group_correlation_results[condition]

        categories = ["Within"]
        mean_values = [group_data["within"]["mean"]]
        sem_values = [group_data["within"]["sem"]]

        for other_cond, stats in group_data["between"].items():
            categories.append(other_cond)
            mean_values.append(stats["mean"])
            sem_values.append(stats["sem"])

        x_positions = np.arange(len(categories))
        bars = ax.bar(x_positions, mean_values, yerr=sem_values, color="skyblue", alpha=0.7, capsize=5)

        # Highlight 'Within' bar
        bars[0].set_color("tomato")

        ax.set_title(f"Condition: {condition}", fontsize=12)
        ax.set_ylabel("Correlation", fontsize=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_ylim([-1, 1])

        for idx, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02, f"{height:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()





# ------------------------------------------------------------------------------------


# ----------------------------- Project Work 4 Functions -----------------------------

# ------------------------------------------------------------------------------------


# ----------------------------- Project Work 5 Functions -----------------------------

# ------------------------------------------------------------------------------------


