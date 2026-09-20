---
name: hvp-continuum-extrapolation
description: Use for the final a_mu global fit and continuum limit.
---
# Continuum Extrapolation
1. Run `continuum_limit.py` testing multiple ansaetze (linear, quadratic, Husung).
2. Use `a_mu-cont_lim-f_pi.py` to aggregate results.
3. Use `model_average-cont_lim.py` to perform AIC/BIC model averaging.
