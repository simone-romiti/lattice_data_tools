""" Routines for input and output to files """

import yaml
import dill

class with_dill:
    @staticmethod
    def dump(obj, path: str):
        dill.dump(obj, open(path, 'wb'))
    #---
    @staticmethod
    def load(path: str):
        return dill.load(open(path, 'rb'))
    #---
#---

class with_yaml:
    @staticmethod
    def load(path: str):
        return yaml.safe_load(open(path, 'r'))
    #---
#---

    
    