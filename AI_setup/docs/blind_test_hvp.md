# HVP Blind Test Protocol

This document defines the "Blind Test" to verify if the `Physics Lead` agent and HVP skills can reproduce high-level analysis results without access to original source scripts.

## 1. The "Clean Room" Environment
The agent will operate in a restricted directory: `scratch/hvp_blind_test/`.
- **Included**: 
    - Raw data: `external_data/HVP_analysis/light-isoQCD/raw_data/`
    - Library: `opencode_ext/lattice_data_tools/`
- **Excluded**:
    - Any files in `light-isoQCD` analysis folders.
    - The `unblinding_process` folder.
    - Original analysis scripts.

## 2. The Challenge
**Task**: Compute the bounded correlator curve and the resulting $a_\mu$ contribution for a single ensemble (e.g., `D96`) using only the generic HVP skills.
**Required Skills**: `hvp-data-io`, `hvp-mistuning`, `hvp-bounding`, `hvp-extrapolation`, `pipeline-state`.

## 3. Validation Metric
**Ground Truth**: The result extracted from the original `light-isoQCD` output files for the chosen ensemble.
**Success Criteria**: The blind agent's result must match the ground truth within a numerical tolerance of $\epsilon = 10^{-4}$.

## 4. Failure Conditions
The test is considered a failure if:
- The agent cannot identify the correct raw data files.
- The agent fails to implement the bounding procedure without reference scripts.
- The result deviates beyond the tolerance.
