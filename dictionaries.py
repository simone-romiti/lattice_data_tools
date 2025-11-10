## routines for nested dictionaries

from typing import Callable, List
from collections import defaultdict

# def nested_dict():
#     """
#     Recursively define defaultdict to support an arbitrary number of levels of nesting
#     This is usefule when the dictionary will have keys assigned in a nested loop
    
#     Example:
#     level_4_dict = nested_dict()
#     level_4_dict['x1']['x2']['x3']['x4'] = "deep_value"

#     ## NOTE: The advantage is that it doesn't cause errors even if we didn't define:
#     # level_4_dict['x1'] = dict({})
#     # level_4_dict['x1']['x2'] = dict({})
#     # ... and so on

#     Returns:
#         defaultdict: structure ready to host a nested dictionary
#     """
#     return defaultdict(nested_dict)
# #---

# def extract_key_combinations(nested_dict: defaultdict, current_keys: List = None) -> List[List]:
#     """ Extract the key combinations for a nested dictionary"""
#     if current_keys is None:
#         current_keys = []
    
#     key_combinations = []

#     for key, value in nested_dict.items():
#         # Create a new list of keys including the current key
#         new_keys = current_keys + [key]
        
#         # If the value is a dictionary, recurse into it
#         if isinstance(value, defaultdict):
#             key_combinations.extend(extract_key_combinations(value, new_keys))
#         else:
#             # If the value is not a dictionary, append the current key combination
#             key_combinations.append(new_keys)

#     return key_combinations
# #---


# def launch_nested_loops(key_lists: List[List], function: Callable, verbose=False):
#     """
#     Launches a nested loop based on the list of lists of string keys.
#     The list of lists can be generared, e.g. with extract_key_combinations()
    
#     Parameters:
#     key_lists (list of lists): A list of lists where each sublist contains string keys.
#     func (lambda): A lambda function to be called with each combination of keys.

#     Example:

#     key_lists = [['A', 'B'], ['X', 'Y', 'Z']]
#     func = lambda key1, key2: print(f"Processing combination: {key1}, {key2}")

#     launch_nested_loops(key_lists, func)
#     """
#     for key_combination in key_lists:
#         if verbose:
#             print(key_combination)
#         #---
#         # Call the lambda function with the current key combination
#         function(*key_combination)
# #-------


class NestedDict(defaultdict):
    def __init__(self, *args, **kwargs):
        # If no default factory is given, use NestedDict itself
        if 'default_factory' not in kwargs and (len(args) == 0 or args[0] is None):
            super().__init__(NestedDict, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
    
    def to_dict(self) -> dict:
        """Convert the nested defaultdict structure into a plain dict."""
        return {k: (v.to_dict() if isinstance(v, NestedDict) else v)
                for k, v in self.items()}
    #---
    def __repr__(self):
        return f"NestedDict({self.to_dict()})"
    #---
    def __getitem__(self, key):
        """
        Extended access:
          - if key is a list/tuple, traverse nested levels
          - else, behave like defaultdict
        """
        if isinstance(key, (list, tuple)):
            d = self
            for k in key:
                d = super(NestedDict, d).__getitem__(k)
            return d
        else:
            return super().__getitem__(key)
    #---
    def __setitem__(self, key, value):
        """
        Extended assignment:
          - if key is a list/tuple, traverse nested levels and assign value
          - else, behave like defaultdict
        """
        if isinstance(key, (list, tuple)):
            d = self
            *parents, last = key
            for k in parents:
                d = super(NestedDict, d).__getitem__(k)
            super(NestedDict, d).__setitem__(last, value)
        else:
            super().__setitem__(key, value)
    #---
    def get_key_combinations(self, parent_keys=None) -> List:
        """
        Recursively extract all key combinations from a nested dictionary.
        
        Args:
            d (dict): Nested dictionary.
            parent_keys (list): Used internally for recursion.
        
        Returns:
            list[list[str]]: List of key paths.
        """
        if parent_keys is None:
            parent_keys = []

        combos = []
        for k, v in self.items():
            new_keys = parent_keys + [k]
            if isinstance(v, NestedDict):
                combos.extend(v.get_key_combinations(new_keys))
            else:
                combos.append(new_keys)
        return combos
    #---
    @staticmethod
    def loop(combinations: List, fun: Callable):
        for combo in combinations:
            fun(combo)
        #---
    #---
            
            
        
