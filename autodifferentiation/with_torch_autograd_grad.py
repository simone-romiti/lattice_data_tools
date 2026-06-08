""" Routines for autodifferentiation using `torch.autograd.grad` """

import torch
import typing

def my_autograd(y: torch.Tensor, x, grad_outputs: torch.tensor, create_graph: bool, retain_graph: bool = False):
    """
    Function to compute autograd:
    - without the hassle of splitting into real and imaginary parts manually
    - with the usual convention: d/dz = d/dx - i d/dy (torch.autograd.grad returns d/dz^{*})

    NOTEs:
    - autograd works only with scalar inputs `y`
    - if `x` is complex, this function returns the Wirtinger derivatives of `y` with respect to the components of `x`:
      `dy/dx = (1/2)*[ dy/d(Re(x)) - i*dy/d(Im(x)) ]`
    """
    y_is_real = not torch.is_complex(y)
    if y_is_real:
        dy_dx = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=create_graph, retain_graph=retain_graph)[0]
    else:
        # NOTE: we have to create the graph necessarily because it is needed for the imaginary part
        Re_dy_dx = torch.autograd.grad(y.real, x, grad_outputs=grad_outputs, create_graph=True, retain_graph=retain_graph)[0]
        Im_dy_dx = torch.autograd.grad(y.imag, x, grad_outputs=grad_outputs, create_graph=create_graph, retain_graph=retain_graph)[0]
        dy_dx = (Re_dy_dx + 1j*Im_dy_dx)
    #---
    if torch.is_complex(x):
        return dy_dx.conj() # autograd returns the Wirtiger derivative with respect to `z^{*}`, not `z`
    else:
        return dy_dx


def directional_derivative_hyperdiagonal(f: typing.Callable, x: torch.Tensor):
    """
     Returns the directional derivative along (1,...,1) of a scalar function f(x):

    $${ \\sum_i \\partial_{x_i} f }$$

    """
    assert(len(x.shape) == 1)
    N = x.numel()
    delta = torch.tensor(0.0, requires_grad=True)

    # basis index -> perturbed input
    def f_i(i):
        e_i = torch.nn.functional.one_hot(i, N).to(x.dtype)
        x_prime = x + delta * e_i
        return f(x_prime)

    # vectorize over all indices 0..N-1
    indices = torch.arange(N)

    sum_f = torch.vmap(f_i)(indices).sum()

    # \\sum_i \\partial_{x_i} f : directional derivative along (1,...,1)
    dir_der = torch.autograd.grad(sum_f, delta, create_graph=True)
    return dir_der


def get_laplacian(f: typing.Callable, x: torch.Tensor):
    """
    Laplacian of a scalar function f(x):

    $${ \\sum_i \\partial_{x_i}^2 f }$$

    This is achieved by differentiating, twice and with respect to the same

    x: torch.tensor with shape (N,)

    """
    assert(len(x.shape) == 1)
    N = x.numel()
    delta = torch.tensor(0.0, requires_grad=True, device=x.device)

    # basis index -> perturbed input
    def f_i(i):
        e_i = torch.nn.functional.one_hot(i, N).to(dtype=x.dtype, device=x.device)
        x_prime = x + delta * e_i
        return f(x_prime)

    # vectorize over all indices 0..N-1
    indices = torch.arange(N)

    sum_f = torch.vmap(f_i)(indices).sum()

    # \\sum_i \\partial_{x_i} f(x) : directional derivative along (1,...,1)
    dir_der = torch.autograd.grad(sum_f, delta, create_graph=True)[0]
    # \\sum_i \\partial_{x_i}^2 f(x)
    laplacian = torch.autograd.grad(dir_der, delta, create_graph=True)[0]
    return laplacian

def get_laplacian_contributions(f: typing.Callable, x: torch.Tensor):
    """
    Contributions to the Laplacian of a scalar function f(x):

    $${ \\partial_{x_i}^2 f }$$

    This is achieved as follows:
    1. df/dx_i = ds/ddelta_i, where
       s=f(x1+delta1,x2,...,x_n)+...+f(x1,x2+delta2,...)+...
    2. d^2f/dx^2 = (d/ddelta_i) (sum_k df/dx_k)

    NOTE: this works because df/dx_i is a sum of derivatives, each evaluated with the arguments shifted where it is needed

    x: torch.tensor with shape (N,)

    """
    assert(len(x.shape) == 1)
    N = x.numel()

    # basis index -> perturbed input
    def f_i(delta_i, i):
        e_i = torch.nn.functional.one_hot(i, N).to(dtype=x.dtype, device=x.device)
        x_prime = x + delta_i * e_i
        return f(x_prime)

    # vectorize over all indices 0..N-1
    indices = torch.arange(N)

    delta = torch.zeros_like(x, requires_grad=True, device=x.device)
    sum_f = torch.vmap(f_i, in_dims=(0,0))(delta, indices).sum()

    # \\partial_{x_i} f(x) : directional derivative along (1,...,1)
    grad_f = torch.autograd.grad(sum_f, delta, create_graph=True)[0]
    # \\sum_i \\partial_{x_i}^2 f(x)
    laplacian = torch.autograd.grad(grad_f.sum(), delta, create_graph=True)[0]
    return laplacian
