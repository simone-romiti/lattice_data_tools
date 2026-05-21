"""
Routines for autodifferentiation.

Some of these functions are used mainly to bechmark the scaling of certain algorithmic choices.
In the rest of the library I use very similar techniques, so this file contains a reference for the methodology.

"""

import typing
import torch

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

