"""
Simple wrapper providing a compiled function in pytorch.
"""

import typing
import torch

def get_compiled_function(f: typing.Callable, *args) -> typing.Callable:
    """
    Call this function passing the function 
    "f" itself and an example list of arguments.

    NOTE: the compiled version of "f" will work only if you pass the same geometry of arguments.
    """
    compiled_f = torch.compile(f) # compilation object (not compiled yet)
    dummy = compiled_f(*args) # triggering compilation
    return compiled_f
    
