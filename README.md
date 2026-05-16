# lattice_data_tools

Lattice Data Tools (LDT) is a `Python` library for [lattice gauge theories](https://en.wikipedia.org/wiki/Lattice_gauge_theory).
It contains programs to produce and analyse lattice data.

Developer: [Simone Romiti](mailto:simone.romiti.1994@gmail.com). 
Feel free to contact me for bug reports, desired features, etc.

## What can I do with LDT?

`lattice_data_tools` provides building blocks for the full analysis pipeline of lattice Quantum Chromodynamics (QCD) simulations. Starting from raw Monte Carlo configurations, the library covers:

- Statistical analysis of Monte Carlo simulations histories (autocorrelation, resampling, etc.)
- Resampling: bootstraps and jackknifes samples.
- Extracting information from lattice QCD correlators:
  - Effective masses, Amplitudes
  - Generalized Eigenvalue Problem (GEVP) for multi-state correlator matrices
- Non-linear least squares fits (support for errors on both x and y)
- Nested Sampling algorithm: production [TODO], analysis
- Model Averaging (Akaike Information Criterion, error budget of systematic effects)
- Frequently made plots in lattice QCD works.
- Lattice Convolutional Neural Networks (L-CNNs)

## Documentation

The documentation can be automatically generated in `./doc` by running the following command:

``` bash
python generate_doc.py \
	--src ./ \
	--out ./doc \
	--github https://github.com/simone-romiti/lattice_data_tools/
```

The information is thaken from each folder and subfolder, looking at their `__init___.py` files, as well as in the docstrings of the various `.py` modules.

## Tests

When I add a new feature to the library I usually write also a test script to test the feature using synthetic data. For those, please refer to the `./test/` folder.
This is however not possible with complicated analyses like for the $g-2$ HVP and HLbL, where one needs also the data.


