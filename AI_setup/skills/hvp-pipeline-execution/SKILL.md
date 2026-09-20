---
name: hvp-pipeline-execution
description: Use when orchestrating the 5-stage HVP analysis pipeline.
---
# HVP Pipeline Execution
Execute the pipeline in this order:
1. `read-blinded_data.py` (Ingestion)
2. `apply-mistunings_ml.py` (Mistuning)
3. `VKVK_eff_curves.py` $\rightarrow$ `VKVK_fit_tail.py` $\rightarrow$ Bounding scripts (Bounding)
4. `apply-UV.py` $\rightarrow$ `volume_interpolation.py` $\rightarrow$ `apply-residual_mistunings.py` (Systematics)
5. `continuum_limit.py` $\rightarrow$ `a_mu-cont_lim-f_pi.py` $\rightarrow$ `model_average-cont_lim.py` (Extrapolation)
