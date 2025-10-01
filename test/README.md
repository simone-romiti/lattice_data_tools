This folder contains test scripts of the routines of the library.

**Available tests**:

- _Monte Carlo history_:
  - `MC_history.py`: Monte Carlo history of some made-up data.

- _Bootstrap sampling_: 
  - `bootstrap_sampling.qmd`: quarto document discussing the bootstrap sampling with examples. Render it with `quarto render bootstrap_sampling.qmd`.
  - `covariance_matrix_bts.py`: covariance matrix extimation from the bootstrap samples

- _Function fitting routines_ :
  * `fit_xyey.py` : function of 1 variable, error on the y only
  * `fit_xiyey.py` : function of 2 variables, error on the y only
  * `fit_xiexiyey.py` : function of 2 variables, error on both x and y

- _Physical point extrapolation_ : `mf_phys.py`

- _Bayesian model averaging_ : 
  - `AIC.py`: Akaike Information Criterion application: assuming continuous distributions
  - `AIC_bootstrap.py`: AIC but with bootstrap samples. The CDF is discontinuos.
- _Reduction of lattice artifacts_: `lambda_method.py`

- _Finite volume effects_:
  - `Luescher_condition.py`: plotting the Luescher's quantization condition
  - `PP_model.py`: Finding the Vector-Vector correlator in the approximation of 2-pions states in a finite volume. We use the Gounaris-Sakurai model in the I=1,J=1 channel.