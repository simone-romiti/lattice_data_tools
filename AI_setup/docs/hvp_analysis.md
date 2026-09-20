# HVP Analysis Framework Notes

This document provides a generic, high-level blueprint for performing Hadronic Vacuum Polarization (HVP) analyses, derived from the `light-isoQCD` project. It is designed to be agnostic of specific dataset paths, focusing instead on the logical pipeline.

## 1. Pipeline Overview

The HVP analysis is a multi-stage pipeline that transforms raw lattice correlators into a final physical observable (e.g., $a_\mu$).

### Stage I: Data Ingestion & Preparation
- **Input**: Binary correlator files ($\text{double-precision}$) and configuration metadata.
- **Process**: 
    - **Parsing**: Read binary files into a structured format (e.g., `NestedDict`).
    - **Bootstrapping**: Generate $N$ bootstrap samples for every data point to propagate statistical uncertainties throughout the chain.
- **Key Output**: A bootstrapped set of correlators $C(t)$ for each ensemble.

### Stage II: Mass Mistuning Correction
Lattice simulations often use slightly "mistuned" quark masses. 
- **Method**: Numerical Differentiation.
- **Process**: 
    - Use correlators computed at multiple valence masses $\mu_1, \mu_2$.
    - Compute the derivative $\frac{\partial C(t)}{\partial \mu}$.
    - Apply a linear shift: $C_{\text{phys}}(t) = C_{\text{sim}}(t) + \frac{\partial C(t)}{\partial \mu} \Delta \mu$.
- **Key Output**: Physical-point correlators.

### Stage III: Tail Treatment (The Bounding Method)
The signal-to-noise ratio of $C(t)$ degrades at large $t$.
- **Process**: 
    - **Effective Mass**: Analyze $m_{\text{eff}}(t)$ to find where the plateau ends.
    - **Bounding**: Replace the noisy tail ($t > t_{\text{cut}}$) with a theoretical model:
        - **Zero Tail**: $C(t) = 0$ for $t > t_{\text{cut}}$.
        - **Exponential Tail**: $C(t) \sim A e^{-mt}$.
        - **Two-Pion Tail**: Use chiral perturbation theory for the long-distance behavior.
- **Key Output**: A "bounded" correlator curve used for integration.

### Stage IV: Integration & Systematic Corrections
- **Integration**: Compute $a_\mu \propto \int_0^\infty K(t) C(t) dt$, where $K(t)$ is the QED kernel.
- **Corrections**:
    - **Finite Volume**: Interpolate/correct for the lattice box size $L$.
    - **UV Correction**: Add perturbative contributions for short-distance behavior.
- **Key Output**: $a_\mu$ values for each lattice spacing $a$.

### Stage V: Final Extrapolation & Model Averaging
- **Continuum Limit**: Fit $a_\mu(a)$ as a function of $a^2$ to extrapolate to $a \to 0$.
- **Model Averaging**: Use AIC/BIC to weight different extrapolation forms (e.g., linear vs. quadratic in $a^2$).
- **Final Output**: $\mathbf{a_\mu^{\text{final}}}$ with a combined statistical and systematic error budget.

---

## 2. Generic Recipe for Independent Execution

To reproduce this analysis on a new dataset:
1. **Identify Correlators**: Locate the $V_k V_k(t)$ binaries.
2. **Define Ensembles**: Create a YAML/JSON config with $a, L, f_\pi$ for each ensemble.
3. **Run Bootstrap**: Generate samples for all input correlators.
4. **Apply Mistunings**: If multi-mass data is available, perform the linear shift.
5. **Bound the Tail**: Determine $t_{\text{cut}}$ via effective mass and apply the bounding model.
6. **Integrate**: Apply the kernel integration to get $a_\mu$ per ensemble.
7. **Extrapolate**: Perform the $a \to 0$ limit using model averaging.
