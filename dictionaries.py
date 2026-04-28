## routines for nested dictionaries

from typing import Callable, List
from collections import defaultdict

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
    def get_key_combinations(self, parent_keys=None, max_depth=None, current_depth=0) -> List:
        """
        Recursively extract all key combinations from a nested dictionary,
        with optional depth limiting.

        Args:
            parent_keys (list): Used internally for recursion.
            max_depth (int, optional): Maximum depth to traverse.
            current_depth (int): Current recursion depth (internal).

        Returns:
            list[list[str]]: List of key paths.
        """
        if parent_keys is None:
            parent_keys = []

        combos = []
        for k, v in self.items():
            new_keys = parent_keys + [k]

            # If we've reached max depth, stop descending
            if max_depth is not None and current_depth >= max_depth - 1:
                combos.append(new_keys)
                continue

            if isinstance(v, NestedDict):
                combos.extend(
                    v.get_key_combinations(
                        new_keys,
                        max_depth=max_depth,
                        current_depth=current_depth + 1
                    )
                )
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
            
            
        
