---
name: hvp-binary-forensics
description: Use when analyzing raw binary correlator formats and headers.
---
# Binary Forensics
When reading binary files:
1. Check if the header is 32-bit or 64-bit.
2. Verify the expected array size against the actual file size.
3. Use `np.fromfile` with the correct `dtype`.
