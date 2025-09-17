""" Routines for input and output to files """

import yaml
import dill

class with_dill:
    @staticmethod
    def dump(obj, path: str):
        assert(path.split('.')[-1] in ['pkl', 'pickle'])
        dill.dump(obj, open(path, 'wb'))
    #---
    @staticmethod
    def load(path: str):
        assert(path.split('.')[-1] in ['pkl', 'pickle'])
        return dill.load(open(path, 'rb'))
    #---
#---

class with_yaml:
    @staticmethod
    def load(path: str):
        assert(path.split('.')[-1] in ['yml', 'yaml'])
        return yaml.safe_load(open(path, 'r'))
    #---
#---

    
    