"""
Routines for autodifferentiation.

Some of these functions are used mainly to bechmark the scaling of certain algorithmic choices.
In the rest of the library I use very similar techniques, so this file contains a reference for the methodology.

"""

import typing
import torch

def my_autograd(y: torch.Tensor, x, create_graph: bool, retain_graph: bool = False):
    """
    Function to computer autograd without the hassle of splitting into real and imaginary parts manually.

    NOTEs:
    - autograd works only with scalar inputs `y`
    - if `x` is complex, this function returns the Wirtinger derivatives of `y` with respect to the components of `x`:
      `dy/dx = (1/2)*[ dy/d(Re(x)) - i*dy/d(Im(x)) ]`
    """
    y_is_real = not torch.is_complex(y)
    if y_is_real:
        dy_dx = torch.autograd.grad(y, x, create_graph=create_graph, retain_graph=retain_graph)[0]
    else:
        # NOTE: we have to create the graph necessarily because it is needed for the imaginary part
        Re_dy_dx = torch.autograd.grad(y.real, x, create_graph=True, retain_graph=retain_graph)[0]
        Im_dy_dx = torch.autograd.grad(y.imag, x, create_graph=create_graph, retain_graph=retain_graph)[0]
        dy_dx = (Re_dy_dx + 1j*Im_dy_dx)
    #---
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
    delta = torch.tensor(0.0, requires_grad=True)

    # basis index -> perturbed input
    def f_i(i):
        e_i = torch.nn.functional.one_hot(i, N).to(x.dtype)
        x_prime = x + delta * e_i
        return f(x_prime)

    # vectorize over all indices 0..N-1
    indices = torch.arange(N)

    sum_f = torch.vmap(f_i)(indices).sum()
    
    # \\sum_i \\partial_{x_i} f(x) : directional derivative along (1,...,1)
    dir_der = torch.autograd.grad(sum_f, delta, create_graph=True)
    # \\sum_i \\partial_{x_i}^2 f(x)
    laplacian = torch.autograd.grad(dir_der, delta, create_graph=True)
    return laplacian

