# fMRI Analysis Pipeline - MATLAB Implementation

This directory contains MATLAB scripts that replicate the Python fMRI analysis pipeline from Assignments 1 and 2. The pipeline performs preprocessing, GLM fitting, statistical analysis, and ROI analysis on fMRI data.

## Overview

The MATLAB implementation includes:
- **Single-subject analysis** (`analyze_single_subject.m`)
- **Multi-subject group analysis** (`analyze_multiple_subjects.m`)
- **Modular helper functions** for each analysis step

## Requirements

### MATLAB Toolboxes
- **Image Processing Toolbox** (for NIfTI file reading)
- Base MATLAB (no SPM required)

### Data Structure
Your project directory should be organized as:
```
project/
├── subj1/
│   ├── bold.nii.gz
│   ├── labels.txt
│   ├── mask4_vt.nii.gz
│   ├── mask8_face_vt.nii.gz
│   └── mask8_house_vt.nii.gz
├── subj2/
│   ├── (same files)
├── hrf.mat
└── matlab/
    ├── analyze_single_subject.m
    ├── analyze_multiple_subjects.m
    ├── (all other .m files)
    └── MATLAB_README.md
```

## Quick Start

### Single-Subject Analysis
```matlab
% Open MATLAB and navigate to the matlab folder
cd /path/to/your/project/matlab

% Run single-subject analysis
analyze_single_subject  % Analyzes subject 1 by default

% To analyze subject 2, edit line 9 in analyze_single_subject.m:
% subject_id = 2;
```

### Multi-Subject Analysis (2 subjects)
```matlab
% Navigate to the matlab folder
cd /path/to/your/project/matlab

% Run group-level analysis
analyze_multiple_subjects  % Automatically analyzes subjects 1 and 2
```

## Pipeline Steps

### 1. Data Loading
- Loads 4D BOLD fMRI data (NIfTI format)
- Loads stimulus labels (conditions and runs)
- Displays data dimensions and metadata

### 2. Anatomical Visualization
- Displays axial, coronal, and sagittal slices
- Middle volume and middle slice by default

### 3. Design Matrix Creation
- Creates binary boxcar functions for each condition
- Excludes 'rest' condition (modeled implicitly)
- Adds run-specific intercepts (12 runs)

### 4. HRF Convolution
- Loads canonical hemodynamic response function
- Convolves each condition with HRF
- Preserves intercepts without convolution

### 5. GLM Fitting
- Fits voxel-wise General Linear Model
- Computes beta coefficients for each condition
- Calculates residuals for statistical testing

### 6. T-Map Computation
- Computes voxelwise t-values: t = β / SE
- SE accounts for residual variance and design covariance
- Generates 4D t-maps [X × Y × Z × N_conditions]

### 7. Contrast Maps
- Defines contrast vectors (e.g., "house > face")
- Computes contrast t-maps
- Visualizes differential activation patterns

### 8. ROI Analysis
- Extracts beta values from predefined ROIs:
  - **Ventral Temporal (VT)**: Broad visual cortex
  - **Face ROI**: Fusiform Face Area
  - **House ROI**: Parahippocampal Place Area
- Computes percent signal change: PSC = (β / baseline) × 100
- Generates bar plots with error bars (SEM)

### 9. Multi-Subject Analysis
- Loops through subjects 1 and 2
- Performs full pipeline for each subject
- Aggregates ROI results across subjects
- Computes group mean and SEM
- Generates group-level visualizations

## Functions Reference

### Data & Design Matrix Functions
| Function | Purpose |
|----------|---------|
| `create_design_matrix.m` | Creates binary design matrix from labels |
| `add_run_intercepts.m` | Adds run-specific baseline columns |
| `convolve_design_matrix.m` | Convolves conditions with HRF |

### GLM & Statistics Functions
| Function | Purpose |
|----------|---------|
| `fit_glm.m` | Fits GLM voxel-by-voxel |
| `compute_t_maps.m` | Computes t-values for beta maps |
| `compute_residual_variance.m` | Computes voxelwise variance |
| `compute_contrast_map.m` | Computes contrast t-maps |

### ROI Analysis Functions
| Function | Purpose |
|----------|---------|
| `compute_roi_percent_signal_change.m` | Extracts PSC for ROIs |
| `plot_roi_results.m` | Visualizes ROI bar graphs |

### Utility Functions
| Function | Purpose |
|----------|---------|
| `redblue.m` | Red-white-blue colormap for t-maps |

## Output Files

### Single-Subject Analysis
- `results_subj1.mat`: Contains beta maps, t-maps, ROI results
- **Figures**: Anatomical slices, design matrix, HRF, t-maps, contrast maps, ROI bar plots

### Multi-Subject Analysis
- `group_results_2subjects.mat`: Contains group mean PSC, SEM, and individual subject data
- **Figures**: Group-level ROI bar plots with SEM error bars

## Example Workflow

```matlab
%% Example: Complete pipeline for 2 subjects

% Step 1: Analyze Subject 1
cd /path/to/project/matlab
analyze_single_subject  % Default: subject_id = 1

% Step 2: Edit analyze_single_subject.m to analyze Subject 2
% Change line 9: subject_id = 2;
analyze_single_subject

% Step 3: Run group analysis
analyze_multiple_subjects

% Step 4: Load and inspect results
load('group_results_2subjects.mat')
group_results.group_mean_psc.Face
group_results.group_sem_psc.Face
```

## Key Differences from Python

| Aspect | Python | MATLAB |
|--------|--------|--------|
| Data loading | `nibabel` | `niftiread()` |
| Array indexing | 0-based | 1-based |
| Matrix operations | NumPy | Native MATLAB |
| Data structures | Pandas DataFrames | Tables & Structs |
| Visualization | Matplotlib | Built-in plotting |

## Customization

### Analyze Different Subjects
Edit `analyze_single_subject.m` line 9:
```matlab
subject_id = 3;  % Change to desired subject
```

Edit `analyze_multiple_subjects.m` line 8:
```matlab
subject_ids = [1, 2, 3, 4];  % Add more subjects
```

### Change Visualization Parameters
In main scripts, modify:
```matlab
slice_to_plot = 28;  % Change slice index
clim([-5 5]);        % Change t-map color limits
```

### Define Custom Contrasts
In `analyze_single_subject.m` (lines 111-116):
```matlab
% Example: scissors > cat
contrast_vector = zeros(size(convolved_matrix, 2), 1);
scissors_idx = find(strcmp(condition_names, 'scissors'));
cat_idx = find(strcmp(condition_names, 'cat'));
contrast_vector(scissors_idx) = 1;
contrast_vector(cat_idx) = -1;
```

### Add More ROIs
In `analyze_single_subject.m` (lines 129-133):
```matlab
% Load additional mask
mask_custom = niftiread('subj1/my_custom_mask.nii.gz') > 0;
roi_masks.CustomROI = mask_custom;
```

## Troubleshooting

### Issue: "Unable to read file"
**Solution**: Ensure NIfTI files are in correct format. MATLAB Image Processing Toolbox supports `.nii` and `.nii.gz`.

### Issue: "Out of memory"
**Solution**: Process one subject at a time, or use smaller ROI masks to reduce computation.

### Issue: "Undefined function or variable"
**Solution**: Ensure all `.m` files are in the same directory or MATLAB path.

### Issue: "Matrix dimensions must agree"
**Solution**: Check that BOLD data and design matrix have matching time dimensions.

## Performance Notes

- **Single-subject analysis**: ~2-5 minutes per subject (depending on hardware)
- **Multi-subject analysis**: ~5-10 minutes for 2 subjects
- **Memory requirements**: ~2-4 GB RAM per subject

## Citation

Based on the Haxby et al. (2001) dataset:
> Haxby, J. V., et al. (2001). Distributed and overlapping representations of faces and objects in ventral temporal cortex. Science, 293(5539), 2425-2430.

## Contact

For questions about this MATLAB implementation, refer to the original Python notebooks (`Assignment1.ipynb`, `Assignment2.ipynb`) or consult `my_functions.py` for algorithm details.

---

**Last Updated**: January 2025
