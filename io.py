""" Routines for input and output to files """

import yaml
import dill

class with_dill:
    @staticmethod
    def load(path: str, verbose=False):
        if verbose:
            print(f"Loading from {path}")
        #---
        assert(path.split('.')[-1] in ['pkl', 'pickle'])
        return dill.load(open(path, 'rb'))
    #---
    @staticmethod
    def dump(obj, path: str, verbose=False):
        if verbose:
            print(f"Saving to {path}")
        #---
        assert(path.split('.')[-1] in ['pkl', 'pickle'])
        dill.dump(obj, open(path, 'wb'))
    #---

class with_yaml:
    @staticmethod
    def load(path: str, verbose=False):
        if verbose:
            print(f"Loading from {path}")
        #---
        assert(path.split('.')[-1] in ['yml', 'yaml'])
        return yaml.safe_load(open(path, 'r'))
    #---

    @staticmethod
    def dump(df, path: str, verbose=False):
        if verbose:
            print(f"Saving to {path}")
        #---
        assert(path.split('.')[-1] in ['yml', 'yaml'])
        yaml.dump(df, open(path, 'w'), default_flow_style=False, indent=4)
    #---

    
    
