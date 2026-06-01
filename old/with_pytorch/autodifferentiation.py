"""
Routines for autodifferentiation.

Some of these functions are used mainly to bechmark the scaling of certain algorithmic choices.
In the rest of the library I use very similar techniques, so this file contains a reference for the methodology.

"""

import typing
import torch
from torch._C import device

from lattice_data_tools.with_pytorch.compilation import get_compiled_function



def my_autograd(y: torch.Tensor, x, create_graph: bool, retain_graph: bool = False):
    """
    Function to compute autograd without the hassle of splitting into real and imaginary parts manually.

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

class WithBruteForce:
    @staticmethod
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



class GradientGenerator:
    """
    Class that generates a (potentially compiled) version or a vectorized calculator of the gradient of a function.
    """
    def __init__(self, f: typing.Callable, x: torch.tensor, do_compile: bool):
        """
        f: function taking as arguments a tensor of shape (n_variables,)
        x: example input of f(x) for each batch: shape==(batchsize,n_variables).
           NOTE:
           it is better (though not always mandatory) to keep the geometry consistent,
           and call the function generated here on arguments with the same geometry

        The gradient is generated as the autodifferentiation with respect to delta=0 of
        s(x,delta) = f(x1+delta,x2,...)+f(x1,x2+delta)+...+f(x1,...,xn+delta)
        """
        assert(len(x.shape) == 2) # (batchsize, n_var)
        self._input_shape = x.shape # saving the expected input shape
        n_var = x.shape[-1] # number of variables for f(x)
        device = x.device
        
        def f_shift(xb, delta, i):
            n = xb.shape[-1]
            ei = (torch.arange(n, device=xb.device) == i).to(xb.dtype)
            return f(xb + delta * ei)

        df = torch.func.grad(f_shift, argnums=1) # abstract gradient object

        # vmap can parallelize only along dimension with the same size
        df_vmapped = torch.func.vmap(
            torch.func.vmap(
                df,
                in_dims=(None, None, 0) # parallelizing only over the variable index --> \\partial_{x_i} f(x)
            ),
            in_dims=(0, None, None),  # parallelizing only over the batch index f(x1^{i}, x2^{i},...)
        )

        delta = torch.tensor(0.0, device=device)
        indices = torch.arange(n_var, device=device)
        uncompiled_gradient_f = lambda x: df_vmapped(x, delta, indices)
        if do_compile:
            self._gradient_function = get_compiled_function(uncompiled_gradient_f, x)
        else:
            self._gradient_function = uncompiled_gradient_f

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def gradient_function(self):
        return self._gradient_function

    
class LaplacianGenerator:
    """
    Class that generates a (potentially compiled) version or a vectorized calculator of the contributions to the Laplacian of a function, i.e.
    $$\\partial_i f(x1,...,xn)$$
    """
    def __init__(self, f: typing.Callable, x: torch.tensor, do_compile: bool):
        """
        f: function taking as arguments a tensor of shape (n_variables,)
        x: example input of f(x) for each batch: shape==(batchsize,n_variables).
           NOTE:
           it is better (though not always mandatory) to keep the geometry consistent,
           and call the function generated here on arguments with the same geometry

        The gradient is generated as the autodifferentiation with respect to delta=0 of
        s(x,delta) = f(x1+delta,x2,...)+f(x1,x2+delta)+...+f(x1,...,xn+delta)
        """
        assert(len(x.shape) == 2) # (batchsize, n_var)
        self._input_shape = x.shape # saving the expected input shape
        n_var = x.shape[-1] # number of variables for f(x)
        device = x.device
        
        def f_shift(xb, delta, i):
            n = xb.shape[-1]
            ei = (torch.arange(n, device=xb.device) == i).to(xb.dtype)
            return f(xb + delta * ei)

        _df = torch.func.grad(f_shift, argnums=1) # abstract gradient object
        d2f = torch.func.grad(_df, argnums=1) # contributions to the Laplacian
        
        # vmap can parallelize only along dimension with the same size
        d2f_vmapped = torch.func.vmap(
            torch.func.vmap(
                d2f,
                in_dims=(None, None, 0) # parallelizing only over the variable index --> \\partial_{x_i}^2 f(x)
            ),
            in_dims=(0, None, None),  # parallelizing only over the batch index f(x1^{i}, x2^{i},...)
        )

        delta = torch.tensor(0.0, device=device)
        indices = torch.arange(n_var, device=device)
        uncompiled_d2f = lambda x: d2f_vmapped(x, delta, indices)
        if do_compile:
            self._d2f_function = get_compiled_function(uncompiled_d2f, x)
        else:
            self._d2f_function = uncompiled_d2f

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def d2f_function(self):
        return self._d2f_function

