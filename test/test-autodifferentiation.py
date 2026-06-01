
from numpy import gradient
import torch

import time
import sys
sys.path.append("../../")

from lattice_data_tools.autodifferentiation.with_torch_func_grad import GradientGenerator, LaplacianGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("Device:", device)


def f(x):
    #return (x ** 2).sum()
    argument = (x * torch.roll(x, shifts=2, dims=0))
    A = torch.tanh(torch.cos(argument)).sum()
    B = torch.exp(-torch.abs(torch.sin((x**2).sum())))
    return A+B


def perf(f, x, N, message):
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(N):
        dummy = f(x+i)
    #---
    torch.cuda.synchronize()
    t1 = time.time()
    print(message, (t1-t0)/N)

n_var = 5*5*3*8
print(f"n_var={n_var}")
torch.manual_seed(20260601)
x = torch.rand(5, n_var, device=device, dtype=torch.float64)

grad = GradientGenerator(f, x, do_compile=False).gradient_function
grad_compiled =  GradientGenerator(f, x, do_compile=True).gradient_function
print("Gradient check:",
      x.shape, grad(x).shape,
      torch.allclose(grad(x), grad_compiled(x))
      )
print("Gradient check:",
      grad(x) - grad_compiled(x)
      )

lapl = LaplacianGenerator(f, x, do_compile=False).d2f_function
lapl_compiled = LaplacianGenerator(f, x, do_compile=True).d2f_function
print("Laplacian check:", x.shape, lapl(x).shape, torch.allclose(lapl(x), lapl_compiled(x)))


N = 10

perf(lambda y: grad(y), x=x, N=N, message="grad (not compiled)")
perf(lambda y: grad_compiled(y), x=x, N=N, message="grad (compiled)")
perf(lambda y: lapl(y), x=x, N=N, message="lapl (not compiled)")
perf(lambda y: lapl_compiled(y), x=x, N=N, message="lapl (compiled)")
