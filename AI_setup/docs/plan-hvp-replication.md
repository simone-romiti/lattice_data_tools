# HVP Replication Plan

Goal: Independently replicate the HVP analysis using reference scripts and data.

## Step 1: Environment Setup
- [ ] Create directory `./scratch/hvp_replication/`
- [ ] Symlink `external_data/HVP_analysis/light-isoQCD/` to the replication dir for easy access.
- [ ] Copy `ensembles.yaml` to the replication dir.

## Step 2: Stage I - Data Ingestion & Bootstrapping
- [ ] Run `read-blinded_data.py` -> Verify binary reading.
- [ ] Run `precompute_kernel.py` -> Verify kernel file creation.
- [ ] Run `gen_bts.py` -> Verify bootstrap samples generation.
- [ ] Run `print-correlator.py` -> Compare output with reference mean/error.

## Step 3: Stage II - Mass Mistuning Correction
- [ ] Run `read-mistunings_ml.py`.
- [ ] Run `apply-mistunings_ml.py` -> Verify shifted correlators.

## Step 4: Stage III - Tail Treatment & Bounding
- [ ] Run `VKVK_eff_curves.py` -> Verify effective mass plateaus.
- [ ] Run `VKVK_fit_tail.py` -> Verify tail fit parameters.
- [ ] Run `bounding-ZeroTail.py`, `bounding-MV_tail.py`, `bounding-2pions.py`.
- [ ] Run `boundings-fits.py` -> Extract $a_\mu$ per ensemble.

## Step 5: Stage IV - Systematic Corrections
- [ ] Run `volume_interpolation.py`.
- [ ] Run `apply-UV.py`.
- [ ] Run `apply-residual_mistunings.py`.

## Step 6: Stage V - Final Extrapolation
- [ ] Run `continuum_limit.py`.
- [ ] Run `model_average-cont_lim.py`.
- [ ] Run `a_mu-fpi_interpolation.py` -> Compare final $a_\mu$ with reference.
