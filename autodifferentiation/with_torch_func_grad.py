"""
Routines for autodifferentiation using `torch.func.grad`.

NOTE: these are separated from the routines using `torch.autograd.grad` because the latter is not supported by `torch.vmap`.

"""

import typing
import torch

def get_compiled_function(f: typing.Callable, *args) -> typing.Callable:
    """
    This function returns a compiled version of the function `f` using `torch.compile`.
    
    Call this function passing the function "f" itself and an example list of arguments.

    NOTE: It is safer to have a compiled version of "f" for each geometry of the arguments tensors. If you change the geometry the function might not work or the performance might be affected.
    """
    compiled_f = torch.compile(f) # compilation object (not compiled yet)
    dummy = compiled_f(*args) # triggering compilation
    return compiled_f


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

