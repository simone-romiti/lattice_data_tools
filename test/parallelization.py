import numpy as np
from joblib import Parallel, delayed
import time

def heavy_compute(x):
    s = 0
    for i in range(10000):
        s += (x * i) % 7
    return s

data = np.arange(10000)  # one task per CPU core

t0 = time.time()
r0 = [heavy_compute(x) for x in data]
t1 = time.time()
print("Not parallel:", t1 - t0, "seconds")

t2 = time.time()
r1 = Parallel(n_jobs=-1)(delayed(heavy_compute)(x) for x in data)
t3 = time.time()
print("Parallel:    ", t3 - t2, "seconds")
