---
name: HVP Data IO
description: Procedures for reading and verifying binary correlator files and managing the HVP data pipeline.
---

1. **Binary Header Verification**:
   - Always read the first 32-64 bytes of a binary file to check the header.
   - Verify that $N_g \times (T/2 + 1) \times N_{\text{avg}}$ matches the total number of doubles in the file.
   - If it doesn't match, the binary format is different from the expected reference; do not proceed with reshaping.

2. **Data Mapping**:
   - Map raw `mix_...` files to the expected `VkVk_...` format using a symlink adapter in `./scratch/hvp_replication/mock_blinded_data`.

3. **Path Management**:
   - Use absolute paths in `ensembles.yaml` to avoid `FileNotFoundError` when running scripts from different working directories.

4. **Validation**:
   - After every data transformation step, print the shape of the resulting arrays to ensure consistency.
