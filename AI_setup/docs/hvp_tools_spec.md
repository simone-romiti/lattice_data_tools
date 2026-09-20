# Technical Specification: HVP Unified Toolset

## Overview
Replace fragmented script execution with a structured controller (`hvp_run.py`) and a diagnostic tool (`hvp_verify.py`).

## 1. `hvp_run.py` (The Controller)

### Class: `HVPController`
- **Purpose**: Orchestrate the 5 stages of the HVP pipeline.
- **Configuration**: Load `ensembles.yaml` to identify target datasets and paths.

### Stage Mapping
| Stage | Description | Reference Script (Example/Placeholder) |
|---|---|---|
| 1 | Data Loading/Preprocessing | `preprocess.py` |
| 2 | Spectral Analysis | `spectral_analysis.py` |
| 3 | Fitting/Extraction | `fitting.py` |
| 4 | Resampling/Error Analysis | `resampling.py` |
| 5 | Final Aggregation/Systematics | `systematics.py` |

### Key Features
- **State Persistence**: 
  - File: `pipeline_state.json`
  - Format: `{ "stage_1": {"status": "success", "timestamp": "..."}, ... }`
  - Logic: Skip completed stages unless `--force` is used.
- **Fast Mode (`--fast`)**:
  - **Constraint**: Do NOT just change a variable like `N_bts`.
  - **Implementation**: In each stage, the controller must pass a `fast_mode=True` flag or the script must slice its input data (e.g., `data = data[:5]`) immediately after loading. All stages must consistently use the same small slice size (e.g., 5) to ensure pipeline compatibility.
- **Environment Setup**:
  - Verify `numba`, `dill`, `scipy`, `numpy` are installed.
  - Set `PYTHONPATH` to include `opencode_ext/lattice_data_tools`.
- **Validation Gates**:
  - After each stage, check for the existence of expected output files (e.g., `.pkl` or `.npy`).
  - Perform a basic sanity check (e.g., file size > 0).

---

## 2. `hvp_verify.py` (The Diagnostic Tool)

### Class: `HVPVerifier`
- **Purpose**: Sanity checks and data audits for pipeline outputs.

### Key Features
- **Data Inspection**:
  - Load `.pkl` files.
  - Report: Tensor shapes, mean, standard deviation, and min/max of the arrays.
- **Covariance Audit**:
  - Compute the condition number of covariance matrices.
  - Warn if $\text{cond}(C) > 10^{12}$ (suggesting `np.linalg.pinv` is required).
- **Consistency Check**:
  - Compare results across different $f_\pi$ points.
  - Flag anomalies where the trend is non-monotonic or deviates significantly from expectations.

## 3. Deployment
- **Location**: `opencode_ext/lattice_data_tools/tools/`
- **Dependencies**: Standard scientific Python stack.
