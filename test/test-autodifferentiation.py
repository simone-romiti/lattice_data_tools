
from numpy import gradient
import torch

import time
import sys
sys.path.append("../../")

from lattice_data_tools.autodifferentiation.with_torch_func_grad import GradientGenerator, LaplacianGenerator, LaplacianGenerator_new
from lattice_data_tools.autodifferentiation.with_torch_autograd_grad import get_laplacian, get_laplacian_contributions, get_laplacian_contributions_h

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("Device:", device)


# def f(x):
#     #return (x ** 2).sum()
#     argument = (x * torch.roll(x, shifts=2, dims=0))
#     A = torch.tanh(torch.cos(argument)).sum()
#     B = torch.exp(-torch.abs(torch.sin((x**2).sum())))
#     return A+B


def g(x, y):
    argument = (x*y * torch.roll(x/y, shifts=2, dims=0))
    A = torch.tanh(torch.cos(argument)).sum()
    B = torch.exp(-torch.abs(torch.sin(((x-y)**2).sum())))
    return A+B

def f(x):
    return g(x,x)


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
B = 1
print(f"n_var={n_var}")
torch.manual_seed(20260601)
x = torch.rand(B, n_var, device=device, dtype=torch.float64)


grad = GradientGenerator(f, x, do_compile=False).gradient_function
grad_compiled =  GradientGenerator(f, x, do_compile=True).gradient_function
print("Gradient check:",
      x.shape, grad(x).shape,
      torch.allclose(grad(x), grad_compiled(x))
      )
print("Gradient check:", torch.allclose(grad(x) , grad_compiled(x)))


N = 20


y = torch.rand(*x.shape, device=x.device, dtype=x.dtype)
lapl_g = LaplacianGenerator_new(g, x, y, do_compile=False).d2f_function
lapl_g_compiled = LaplacianGenerator_new(g, x, y, do_compile=True).d2f_function
perf(lambda z: lapl_g(z,y), x=x, N=N, message="lapl_g (not compiled)")
perf(lambda z: lapl_g_compiled(z,y), x=x, N=N, message="lapl_g (compiled)")

print(lapl_g(x, y).shape)

lapl = LaplacianGenerator_new(f, x, do_compile=False).d2f_function
lapl_compiled = LaplacianGenerator_new(f, x, do_compile=True).d2f_function
print("Laplacian check:", x.shape, lapl(x).shape, torch.allclose(lapl(x), lapl_compiled(x)))

lapl_autograd = torch.stack([get_laplacian(f=f, x=x[b,:]) for b in range(B)], dim=0)
lapl_contribs_autograd = torch.stack([get_laplacian_contributions(f=f, x=x[b,:]) for b in range(B)], dim=0)


perf(lambda y: torch.stack([get_laplacian_contributions(f=f, x=y[b,:]) for b in range(B)], dim=0), x=x, N=N, message="lapl (autograd)")

perf(lambda y: grad(y), x=x, N=N, message="grad (not compiled)")
perf(lambda y: grad_compiled(y), x=x, N=N, message="grad (compiled)")
perf(lambda y: lapl(y), x=x, N=N, message="lapl (not compiled)")
perf(lambda y: lapl_compiled(y), x=x, N=N, message="lapl (compiled)")

