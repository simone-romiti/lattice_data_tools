# lattice_data_tools

**A Python library for the analysis of lattice QCD data.**

Developed by [Simone Romiti](mailto:simone.romiti.1994@gmail.com).

---

## What does it do?

`lattice_data_tools` provides building blocks for the full analysis pipeline of lattice Quantum Chromodynamics (QCD) simulations. Starting from raw Monte Carlo configurations, the library covers:

| What | Module |
|---|---|
| **Resampling** — bootstrap and jackknife with auto-binning for correlated data | `bootstrap.py`, `jackknife.py` |
| **Autocorrelation analysis** — Gamma method, integrated τ_int, optimal binning | `uwerr.py` |
| **Effective curves** — effective mass / amplitude from correlators (log, cosh, sinh strategies) | `effective_curves.py` |
| **GEVP** — Generalized Eigenvalue Problem for multi-state correlator matrices | `gevp.py` |
| **Function fitting** — trajectory method for f: ℝⁿ → ℝᵐ with errors on x and y | `fit/` |
| **Model averaging** — AIC, AICc, BIC, AIC_Ncut; statistical vs. systematic error decomposition | `model_averaging/` |
| **Nested sampling** — phase-space weights, partition function curves, density of states | `nested_sampling/` |
| **Lattice artifacts** — λ-method for combining different regularizations towards the continuum | `artifacts.py` |
| **Physical point** — sequential tuning of quark masses (Mπ → mℓ, MK → ms, MD_s → mc) | `physical_point/` |
| **Muon g−2 HVP** — QED kernel, time windows, Gounaris–Sakurai model, finite-volume corrections | `gm2/HVP/` |
| **Plotting** — matplotlib (PDF), plotly (interactive HTML), plotnine (ggplot-style) | `plotting/` |

---

## Highlights

### Bootstrap and Jackknife

`BootstrapSamples` and `JackknifeSamples` are NumPy array subclasses that carry both the original mean and the resampled distribution. They actively guard against common mistakes:

```python
from lattice_data_tools.bootstrap import uncorrelated_confs_to_bts, BootstrapSamples
import numpy as np

data = np.random.normal(loc=5.0, scale=1.0, size=500)
bts  = uncorrelated_confs_to_bts(data, N_bts=200)

print(bts.unbiased_mean())   # mean of original sample
print(bts.error())           # standard error (bootstrap estimate)
np.mean(bts)                 # TypeError! Use bts.mean() instead
```

For correlated Monte Carlo chains, `correlated_confs_to_bts` handles automatic binning via the integrated autocorrelation time τ_int.

### Fitting

The trajectory method fits f: ℝⁿ → ℝᵐ by minimising the distance in n+m dimensional space, supporting errors on both x and y and full covariance matrices:

```python
from lattice_data_tools.fit.xyey import fit_xyey, polynomial_fit_xyey

res = fit_xyey(ansatz=lambda x, p: p[0]*np.exp(-p[1]*x),
               x=t, y=C, ey=dC, guess=[1.0, 0.3])
print(res["par"], "chi2/dof =", res["ch2_dof"])
```

For polynomial fits, the exact closed-form solution is used:

```python
res = polynomial_fit_xyey(N_deg=2, x=x, y=y, ey=ey)
```

### Model Averaging (AIC / BIC)

Combines multiple fit results into a single estimate with both statistical and systematic uncertainties:

```python
from lattice_data_tools.model_averaging.IC import get_weights, with_CDF

w  = get_weights(ch2=ch2, n_par=n_par, n_data=n_data, IC="AIC")
res = with_CDF.get_P(y=[fit_values_per_model], w=w)
q  = with_CDF.get_quantiles(y=res["y"], P=res["P"])
print(q["50%"], "±", (q["84%"] - q["16%"]) / 2)
```

### Effective Mass

```python
from lattice_data_tools.effective_curves import get_m_eff, fit_eff_mass

m_eff = get_m_eff(C, strategy="cosh", T=48)
m0    = fit_eff_mass(m_eff[tmin:tmax], dm_eff[tmin:tmax])
```

---

## Installation

The library is not yet on PyPI. Clone and use directly:

```bash
git clone <repo-url>
pip install numpy scipy matplotlib plotly plotnine joblib pandas dill pyyaml
```

Then import as:
```python
import lattice_data_tools
```
or add the parent directory to your `PYTHONPATH`.

---

## Tests

```bash
pip install pytest
pytest test/ -v
```

---

## Documentation

The full API documentation is generated with [Quarto](https://quarto.org) and lives in `doc/`. To build it locally:

```bash
cd doc/
quarto render
```

---

## Contributing / Contact

Bug reports and suggestions: [simone.romiti.1994@gmail.com](mailto:simone.romiti.1994@gmail.com)
