## routines for nested dictionaries

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
####