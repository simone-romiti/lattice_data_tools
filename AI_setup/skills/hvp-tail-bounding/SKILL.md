---
name: hvp-tail-bounding
description: Use when replacing noisy tails of correlators.
---
# Tail Bounding
SOP:
1. Use `VKVK_eff_curves.py` to find the $t_{cut}$ where signal-to-noise degrades.
2. Apply `ZeroTail` (lower bound) and `2pions` (upper bound).
3. Use `MV_tail` for the central value.
4. Use `np.linalg.pinv` for covariance matrices with high condition numbers.
