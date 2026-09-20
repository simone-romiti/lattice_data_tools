# Plan: Unified HVP Toolset Implementation

- [x] Define `HVPController` class structure for `hvp_run.py`.
- [x] Define `HVPVerifier` class structure for `hvp_verify.py`.
- [x] Specify the mapping of 5 stages to scripts.
- [x] Design the state persistence format (`pipeline_state.json`).
- [x] Design the `--fast` mode data slicing mechanism.
- [x] Create a technical specification for the coder in `./scratch/hvp_tools_spec.md`.

## 2. Implementation (Coder Delegation)
- [x] Create directory `opencode_ext/lattice_data_tools/tools/`.
- [x] Implement `hvp_run.py` based on the spec.
- [x] Implement `hvp_verify.py` based on the spec.
- [x] Verify `ensembles.yaml` integration.

## 3. Validation & Integration (Architect/Foreman)
- [ ] Verify that `hvp_run.py` correctly handles state and `--fast` mode.
- [ ] Verify that `hvp_verify.py` provides meaningful diagnostics.
- [ ] Update `routing-rules.md` or agent skills if these tools create new common patterns.


## 3. Validation & Integration (Architect/Foreman)
- [ ] Verify that `hvp_run.py` correctly handles state and `--fast` mode.
- [ ] Verify that `hvp_verify.py` provides meaningful diagnostics.
- [ ] Update `routing-rules.md` or agent skills if these tools create new common patterns.
