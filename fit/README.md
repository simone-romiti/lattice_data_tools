This folder contains scripts with fitting routines.

**Backbone routines**: `./trajectory.py`. Fitting a trajectory in a $(n+m)$-dimensional space, with a constraint determined by a function $f: \mathbb{R}^n \to \mathbb{R}^m$: 

$$y_i = f_i(\vec{x})$$ 

**Supported types of fit**:

- `xyey.py`: $f: \mathbb{R} \to \mathbb{R}$ with errors on the $y$ only
- `xiyey.py`: $f: \mathbb{R}^n \to \mathbb{R}$ with errors on the $y$ only
- `xiexiyey.py`: $f: \mathbb{R}^n \to \mathbb{R}$ with errors on both the $x_i$ and $y$
- `xiexiyieyi.py`: $f: \mathbb{R}^n \to \mathbb{R}^m$ with errors on both the $x_i$ and $y_i$. This is an alias for the routine for fitting a trajectory.

**Tests can be found in the `test/` subfolder of the main directory.