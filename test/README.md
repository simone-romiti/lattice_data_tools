This folder contains test scripts of the routines of the library.

**Available tests**:

- _Bootstrap sampling_: 
  - `bootstrap_sampling.qmd`: quarto document discussing the bootstrap sampling with examples. Render it with `quarto render bootstrap_sampling.qmd`.

- _Function fitting routines_ :
  * `fit_xyey.py` : function of 1 variable, error on the y only
  * `fit_xiyey.py` : function of 2 variables, error on the y only
  * `fit_xiexiyey.py` : function of 2 variables, error on both x and y

- _Physical point extrapolation_ : `mf_phys.py`

- _Bayesian model averaging_ : 
  - `AIC.py`: Akaike Information Criterion application: assuming continuous distributions
  - `AIC_bootstrap.py`: AIC but with bootstrap samples. The CDF is discontinuos.
- _Reduction of lattice artifacts_: `lambda_method.py`

