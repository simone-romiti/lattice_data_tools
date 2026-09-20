# HVP Replication Playbook

This playbook describes the manual replication of the HVP analysis pipeline.

## Pipeline Overview
The pipeline transforms raw binary correlators into the final $a_\mu^{\text{HVP}}$ contribution.

| Stage | Script | Input | Output | Key Check |
|-------|---------|--------|--------|-----------|
| I     | `ingest_data.py` | Binary files | `correlators.npy` | Header size & shape |
| II    | `apply-mistunings_ml.py` | `correlators.npy` | `mistuned.npy` | Broadcast shapes |
| III   | `tail_treatment.py` | `mistuned.npy` | `bounded.npy` | $t_{cut}$ stability |
| IV    | `systematics_corrections.py`| `bounded.npy` | `corrected.npy` | UV/Vol check |
| V     | `continuum_extrap.py` | `corrected.npy` | Final Value | AIC weights |

## Critical Verification Steps
1. **Binary Verification**: Before Stage I, use `hvp-binary-forensics` to ensure files are not corrupted.
2. **Stability Check**: In Stage III, ensure `np.linalg.pinv` is used.
3. **Reference Match**: After each stage, compare the output `.npy` or scalar value with the reference files in `external_data/HVP_analysis/reference_results/`.

## Troubleshooting
- **`ValueError` during reshape**: Check the binary header; you might be using a 32-bit offset for 64-bit data.
- **`LinAlgError` during fit**: The covariance matrix is singular. Use `pinv`.
- **Broadcasting Error in Mistuning**: Check that the `ensembles.yaml` mapping matches the input array dimensions.
