## routines for nested dictionaries

from typing import Callable, List
from collections import defaultdict

def nested_dict():
    """
    Recursively define defaultdict to support an arbitrary number of levels of nesting
    This is usefule when the dictionary will have keys assigned in a nested loop
    
    Example:
    level_4_dict = nested_dict()
    level_4_dict['x1']['x2']['x3']['x4'] = "deep_value"

    ## NOTE: The advantage is that it doesn't cause errors even if we didn't define:
    # level_4_dict['x1'] = dict({})
    # level_4_dict['x1']['x2'] = dict({})
    # ... and so on

    Returns:
        defaultdict: structure ready to host a nested dictionary
    """
    return defaultdict(nested_dict)
#---

def extract_key_combinations(nested_dict: defaultdict, current_keys: List = None) -> List[List]:
    """ Extract the key combinations for a nested dictionary"""
    if current_keys is None:
        current_keys = []
    
    key_combinations = []

    for key, value in nested_dict.items():
        # Create a new list of keys including the current key
        new_keys = current_keys + [key]
        
        # If the value is a dictionary, recurse into it
        if isinstance(value, defaultdict):
            key_combinations.extend(extract_key_combinations(value, new_keys))
        else:
            # If the value is not a dictionary, append the current key combination
            key_combinations.append(new_keys)

    return key_combinations
#---


def launch_nested_loops(key_lists: List[List], function: Callable, verbose=False):
    """
    Launches a nested loop based on the list of lists of string keys.
    The list of lists can be generared, e.g. with extract_key_combinations()
    
    Parameters:
    key_lists (list of lists): A list of lists where each sublist contains string keys.
    func (lambda): A lambda function to be called with each combination of keys.

    Example:

    key_lists = [['A', 'B'], ['X', 'Y', 'Z']]
    func = lambda key1, key2: print(f"Processing combination: {key1}, {key2}")

    launch_nested_loops(key_lists, func)
    """
    for key_combination in key_lists:
        if verbose:
            print(key_combination)
        #---
        # Call the lambda function with the current key combination
        function(*key_combination)
#-------


class NestedDict(defaultdict):
    def __init__(self):
        super().__init__(NestedDict)
    #---
    def to_dict(self) -> dict:
        """Convert the nested defaultdict structure into a plain dict."""
        return {k: (v.to_dict() if isinstance(v, NestedDict) else v)
                for k, v in self.items()}
    #---
    def __repr__(self):
        return f"NestedDict({self.to_dict()})"
    #---
    @staticmethod
    def key_combinations(nested_dict: defaultdict, current_keys: List = None) -> List[List]:
        """ Extract the key combinations for a nested dictionary"""
        if current_keys is None:
            current_keys = []
        
        key_combinations = []

        for key, value in nested_dict.items():
            # Create a new list of keys including the current key
            new_keys = current_keys + [key]
            
            # If the value is a dictionary, recurse into it
            if isinstance(value, defaultdict):
                key_combinations.extend(extract_key_combinations(value, new_keys))
            else:
                # If the value is not a dictionary, append the current key combination
                key_combinations.append(new_keys)

        return key_combinations
    #---
    @staticmethod
    def loop_over_key_combinations(key_combinations: List[List], fun: Callable, verbose=False) -> None:
        """
        Launches a nested loop based on the list of lists of string keys.
        The list of lists can be generared, e.g. with extract_key_combinations()
        
        Parameters:
        key_lists (list of lists): A list of lists where each sublist contains string keys.
        func (lambda): A lambda function to be called with each combination of keys.

        Example:

        key_lists = [['A', 'B'], ['X', 'Y', 'Z']]
        func = lambda key1, key2: print(f"Processing combination: {key1}, {key2}")

        launch_nested_loops(key_lists, func)
        """
        assert len({len(kc) for kc in key_combinations}) <= 1 and len(key_combinations) >= 0, "Illegal loop over key combinations for NestedDict. They don't have all the same length"
        for key_combination in key_combinations:
            if verbose:
                print(key_combination)
            #---
            # Call the lambda function with the current key combination
            fun(*key_combination)
        #---
        return None
    #---
