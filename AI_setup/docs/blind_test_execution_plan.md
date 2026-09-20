# Execution Plan for HVP Blind Test

## Step 1: Environment Initialization
Run the setup script to create the Clean Room.
```bash
./scripts/setup_hvp_blind_test.sh
```

## Step 2: Agent Deployment
Trigger the `Physics Lead` agent with the following prompt:
"You are in a clean room environment at `scratch/hvp_blind_test/`. Using only the `lattice_data_tools` library and the raw data provided, compute the bounded correlator and $a_\mu$ for the D96 ensemble. Use the HVP skills for data IO, mistuning, and bounding. Document your process in `scratch/hvp_blind_test/analysis_log.md`."

## Step 3: Validation
The `Codebase Analyst` agent must:
1. Read the original ground truth from the `light-isoQCD` output.
2. Read the result produced by the blind agent in `scratch/hvp_blind_test/`.
3. Compare the values and report the absolute difference.

## Step 4: Final Report
Generate a pass/fail report based on the tolerance $\epsilon = 10^{-4}$.
