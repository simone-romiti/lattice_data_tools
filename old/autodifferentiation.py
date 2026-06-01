"""
Routines for autodifferentiation.

Some of these functions are used mainly to bechmark the scaling of certain algorithmic choices.
In the rest of the library I use very similar techniques, so this file contains a reference for the methodology.

"""

import typing
import torch
import dill
import os

import sys
sys.path.append('../')
from lattice_data_tools.io import with_dill
from lattice_data_tools.with_pytorch import get_compiled_function


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

class with_autograd:
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

    
# if __name__ == "__main__":
#     import time
#     import os
#     import torch
#     import torch.func
#     from torch.fx.experimental.proxy_tensor import make_fx

#     device_name = "cuda"  # change to "cpu" if no GPU
#     device_name = "cpu"
#     device = torch.device(device_name)
#     print("Device", device)

#     # ------------------------------------------------------------------ #
#     #  Functions                                                           #
#     # ------------------------------------------------------------------ #
#     def f0(x):
#         return (x ** 2).sum()

#     def f(x, delta, i):
#         n = x.shape[-1]
#         # torch.arange uses x.device so vmap traces it correctly at runtime
#         ei = (torch.arange(n, device=x.device) == i).to(x.dtype)
#         return f0(x + delta * ei)

#     # ------------------------------------------------------------------ #
#     #  Data                                                                #
#     # ------------------------------------------------------------------ #
#     B = 5
#     n_var = 2000
#     x = torch.rand(B, n_var, device=device)
#     delta = torch.tensor(0.0, device=device)
#     indices = torch.arange(n_var, device=device)   # pre-allocated once

#     # Quick sanity check
#     f0_values = torch.func.vmap(f0, in_dims=(0,))(x)
#     print("f0 vmap matches manual:", torch.allclose(f0_values, (x ** 2).sum(dim=1)))

#     # ------------------------------------------------------------------ #
#     #  Build grad transforms                                               #
#     # ------------------------------------------------------------------ #
#     df_ddelta    = torch.func.grad(f,         argnums=1)
#     d2f_ddelta2  = torch.func.grad(df_ddelta, argnums=1)

#     # Eager vmapped second derivative — used as the benchmark baseline
#     d2f_ddelta2_vmapped = torch.func.vmap(
#         torch.func.vmap(d2f_ddelta2, in_dims=(None, None, 0)),
#         in_dims=(0, None, None),
#     )

#     # ------------------------------------------------------------------ #
#     #  Helpers                                                             #
#     # ------------------------------------------------------------------ #
#     def sync():
#         """Synchronise CUDA stream; no-op on CPU."""
#         if device_name == "cuda":
#             torch.cuda.synchronize()

#     def patch_graph_device(gm, target_device):
#         """
#         make_fx bakes the tracing device into every `device=` kwarg in the
#         graph (e.g. the aten.arange call inside the traced vmap body).
#         When the model is later run on a *different* device (e.g. you traced
#         on CPU then run on CUDA, or you reload a CPU-traced export on CUDA)
#         those baked-in ops still execute on the original device, forcing
#         implicit host<->device copies on every call and killing any speedup.

#         This function patches all device= kwargs in the graph to
#         `target_device` and drops `pin_memory=True` (invalid on CUDA).
#         It must be called both after make_fx (before export) and after
#         torch.export.load (before torch.compile), because the device string
#         survives serialisation.
#         """
#         td = torch.device(target_device)
#         for node in gm.graph.nodes:
#             new_kwargs = dict(node.kwargs)
#             if "device" in new_kwargs and isinstance(new_kwargs["device"], torch.device):
#                 new_kwargs["device"] = td
#             if "pin_memory" in new_kwargs and td.type == "cuda":
#                 new_kwargs["pin_memory"] = False
#             node.kwargs = new_kwargs
#         gm.graph.lint()
#         gm.recompile()
#         return gm

#     class TracedWrapper(torch.nn.Module):
#         """Thin nn.Module shell around a make_fx GraphModule for torch.export."""
#         def __init__(self, gm):
#             super().__init__()
#             self.gm = gm

#         def forward(self, x, delta, indices):
#             return self.gm(x, delta, indices)

#     # ------------------------------------------------------------------ #
#     #  Export / load                                                       #
#     # ------------------------------------------------------------------ #
#     # We always compile with torch.compile regardless of which branch runs.
#     # The export file lets us skip the slow make_fx tracing on subsequent
#     # runs, but torch.compile must still be applied after loading so that
#     # Triton / inductor kernels are generated (the exported file stores the
#     # ATen graph, NOT compiled kernels).

#     model_path = f"./model-{device_name}.pt2"

#     if not os.path.exists(model_path):
#         print("First run: tracing, exporting, and compiling …")

#         # 1. Lower vmap+grad to a flat ATen graph (torch.export cannot trace
#         #    through torch.func transforms directly).
#         traced_fn = make_fx(d2f_ddelta2_vmapped)(x, delta, indices)

#         # 2. FIX: patch device= kwargs so the graph runs on the correct device
#         #    both now and after reloading on a potentially different run.
#         patch_graph_device(traced_fn, device_name)

#         module_to_export = TracedWrapper(traced_fn)

#         # 3. Export and save (captures the patched ATen graph).
#         exported = torch.export.export(
#             module_to_export,
#             args=(x, delta, indices),
#         )
#         torch.export.save(exported, model_path)
#         print(f"Saved to {model_path}")

#         # 4. Compile the module we just built (reuse it, don't reload).
#         d2f_ddelta2_compiled = torch.compile(module_to_export)

#     else:
#         print(f"Subsequent run: loading from {model_path} …")

#         loaded_ep = torch.export.load(model_path)
#         gm = loaded_ep.module()   # plain GraphModule, not yet compiled

#         # FIX: patch device= kwargs again — they survive serialisation and
#         # would cause CPU<->GPU syncs if left pointing at the trace device.
#         patch_graph_device(gm, device_name)

#         # FIX: wrap in torch.compile so inductor/Triton kernels are generated.
#         # Without this the graph runs interpreted and matches eager speed.
#         d2f_ddelta2_compiled = torch.compile(gm)

#     # ------------------------------------------------------------------ #
#     #  Warm-up (let torch.compile finish JIT compilation before timing)   #
#     # ------------------------------------------------------------------ #
#     if not os.path.exists("compiled_function.pkl"):
#         print("Warming up …")
#         sync()
#         t0 = time.time()
#         _ = d2f_ddelta2_compiled(x, delta, indices)
#         sync()
#         t1 = time.time()
#         print("warmup :", (t1 -t0))
#         with_dill.dump(d2f_ddelta2_compiled, "compiled_function.pkl")
#     else:
#         d2f_ddelta2_compiled = with_dill.load("./compiled_function.pkl")

#     # ------------------------------------------------------------------ #
#     #  Correctness check                                                   #
#     # ------------------------------------------------------------------ #
#     result   = d2f_ddelta2_compiled(x, delta, indices)
#     expected = d2f_ddelta2_vmapped(x, delta, indices)
#     print("Correctness check:", torch.allclose(result, expected, atol=1e-5))

#     # ------------------------------------------------------------------ #
#     #  Benchmark                                                           #
#     # ------------------------------------------------------------------ #
#     N = 10

#     # Compiled
#     sync()
#     t0 = time.time()
#     for _ in range(N):
#         _ = d2f_ddelta2_compiled(x, delta, indices)
#     sync()
#     t1 = time.time()
#     compiled_ms = (t1 - t0) / N 

#     # Eager baseline
#     sync()
#     t2 = time.time()
#     for _ in range(N):
#         _ = d2f_ddelta2_vmapped(x, delta, indices)
#     sync()
#     t3 = time.time()
#     eager_ms = (t3 - t2) / N 

#     print(f"compiled avg over {N} calls: {compiled_ms:.3f} ms")
#     print(f"eager    avg over {N} calls: {eager_ms:.3f} ms")
#     print(f"speedup: {eager_ms / compiled_ms:.2f}x")

def GradientComponentsGenerator:
    """
    Class that generates a (potentially compiled) version or a vectorized calculator of the gradient of a function.
    """
    def __init__(self, f: typing.Callable, x: torch.tensor):
        """
        f: function taking as arguments a tensor of shape (n_variables,)
        x: example input of f(x), shape==(batchsize,n_variables).
           NOTE:
           you need to keep the geometry consistent,
           and call the function generated here on arguments with the same geometry

        The gradient is generated as the autodifferentiation with respect to delta=0 of
        s(x,delta) = f(x1+delta,x2,...)+f(x1,x2+delta)+...+f(x1,...,xn+delta)
        """
        self._input_shape = x.shape
        
        def f_shift(x, delta, i):
            n = x.shape[-1]
            ei = (torch.arange(n, device=x.device) == i).to(x.dtype)
            return f0(x + delta * ei)

        df = torch.func.grad(f_shift, argnums=1) # abstract gradient object

        # vmap can parallelize only along dimension with the same size
        df_vmapped = torch.func.vmap(
            torch.func.vmap(
                df,
                in_dims=(None, None, 0) # parallelizing only over the variable index --> \\partial_{x_i} f(x)
            ),
            in_dims=(0, None, None),  # parallelizing only over the batch index f(x1^{i}, x2^{i},...)
        )

        self.df_compiled = get_compiled_function(df_vmapped, x)

    @property
    def input_shape(self):
        return self._input_shape
    
    def gradient_function(self):
        return self.df_compiled

if __name__ == "__main__":
    import time
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("Device:", device)

    # ------------------------------------------------------------
    # Function
    # ------------------------------------------------------------
    def f0(x):
        return (x ** 2).sum()

    def f(x, delta, i):
        n = x.shape[-1]
        ei = (torch.arange(n, device=x.device) == i).to(x.dtype)
        return f0(x + delta * ei)

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------
    n_var = 5*5*5*3*8
    print(f"n_var={n_var}")
    x = torch.rand(5, n_var, device=device)
    delta = torch.tensor(0.0, device=device)
    indices = torch.arange(n_var, device=device)

    # ------------------------------------------------------------
    # Autograd setup
    # ------------------------------------------------------------
    df = torch.func.grad(f, argnums=1)
    d2f = torch.func.grad(df, argnums=1)

    d2f_vmapped = torch.func.vmap(
        torch.func.vmap(d2f, in_dims=(None, None, 0)),
        in_dims=(0, None, None),
    )

    # ------------------------------------------------------------
    # Compile version
    # ------------------------------------------------------------
    compiled = torch.compile(d2f_vmapped)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ============================================================
    # EAGER baseline (IMPORTANT — this is what you were missing)
    # ============================================================
    print("Running eager baseline...")

    sync()
    t0 = time.time()
    eager_out = d2f_vmapped(x, delta, indices)
    sync()
    t1 = time.time()
    eager_time = t1 - t0
    print("eager time:", eager_time)
    
    # ------------------------------------------------------------
    # Warmup compiled (first run = compilation)
    # ------------------------------------------------------------
    print("Warmup compiled...")

    sync()
    t0 = time.time()
    compiled_out = compiled(x+1, delta, indices)
    sync()
    t1 = time.time()

    print("warmup (compile + run):", t1 - t0)

    # ------------------------------------------------------------
    # Correctness check
    # ------------------------------------------------------------
    print("Correctness:", torch.allclose(eager_out, compiled_out, atol=1e-5))

    # ------------------------------------------------------------
    # Benchmark compiled
    # ------------------------------------------------------------
    N = 10

    sync()
    t0 = time.time()
    for _ in range(N):
        compiled(x, delta, indices)
    sync()
    t1 = time.time()

    compiled_time = (t1 - t0) / N

    print(f"compiled avg: {compiled_time:.6f} s")
    print(f"eager avg:    {eager_time:.6f} s")
    print(f"speedup:      {eager_time / compiled_time:.2f}x")
    
    
