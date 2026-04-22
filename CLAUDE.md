# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`lattice_data_tools` is a research Python library for lattice QCD data analysis, focused on statistical methods, function fitting, and domain-specific computations (e.g., muon g-2 HVP).

## Running Tests

Tests live in `test/` and must be run from that directory:

```bash
cd test
python fit_xyey.py
python AIC.py
python bootstrap_sampling.py
# etc.
```

Quarto-based test/tutorial documents are rendered with:

```bash
quarto render bootstrap_sampling.qmd
```

## Architecture

### Core Data Types

- **`bootstrap.py`** — `BootstrapSamples(np.ndarray)`: first axis index 0 is the unbiased mean; indices 1+ are bootstrap replicates. Blocks `np.mean`/`np.std` to prevent misuse. Use `.mean()`, `.error()`, `.unbiased_mean()`, `.covariance_matrix()`.
- **`jackknife.py`** — `JackknifeSamples(np.ndarray)`: analogous subclass for jackknife resampling.
- **`dictionaries.py`** — `NestedDict`: defaultdict subclass supporting multi-level key access via lists/tuples.

### Fitting Framework (`fit/`)

The backbone is `fit/trajectory.py`, implementing a maximum-likelihood trajectory method for fitting f: ℝⁿ → ℝᵐ with errors on both x and y. Specialized wrappers:

| File | Use case |
|---|---|
| `fit/xyey.py` | 1D → 1D, y-errors only |
| `fit/xiyey.py` | nD → 1D, y-errors only |
| `fit/xiexiyey.py` | nD → 1D, x and y errors |
| `fit/xiexiyieyi.py` | nD → mD (general trajectory) |

### Statistical Methods

- **`bootstrap.py`**: `uncorrelated_confs_to_bts()`, `correlated_confs_to_bts()`, `auto_binning()`, `parametric_gaussian_bts()`
- **`jackknife.py`**: `uncorrelated_confs_to_jkf()`, `correlated_confs_to_jkf()`
- **`uwerr.py`**: Gamma Method for autocorrelation time and error estimates

### Domain-Specific Modules

- **`effective_curves.py`**: Effective mass (`get_m_eff_log`, `get_m_eff_bkw`) and amplitude extraction; `fit_eff_mass()`.
- **`gevp.py`**: Generalized Eigenvalue Problem for multi-level correlator analysis.
- **`artifacts.py`**: `lambda_method` class — reduces lattice discretization effects by combining different regularizations.
- **`model_averaging/`**: Bayesian model averaging via `IC.py` (information criteria) and `with_bts.py` (bootstrap-based).
- **`nested_sampling/`**: Analysis of nested sampling results — weights, phase space, partition function curves (`weights.py`).
- **`physical_point/tuning_mf.py`**: Sequential tuning to physical quark masses (Mπ → mℓ, MK → ms, MD_s → mc).
- **`gm2/HVP/`**: Muon g-2 HVP computation — QED kernel (`kernel.py`), main calculation (`amu.py`), Gounaris-Sakurai resonance model, finite-volume corrections, time windows.

### Utilities

- **`constants.py`**: Physical constants (masses in MeV, unit conversions fm↔MeV⁻¹).
- **`io.py`**: File I/O via `dill` (pickle) and YAML.
- **`plotting/`**: Three backends — `with_matplotlib/` (PDF), `with_plotly/` (interactive HTML), `with_plotnine/` (ggplot-style, experimental).

## Key Dependencies

`numpy`, `scipy`, `matplotlib`, `plotly`, `plotnine`, `joblib`, `pandas`, `dill`, `pyyaml`
