import os

class Paths(object):
    root = "H:\\vision\\cephalometry\\datasets"
    
    @classmethod
    def dataset_root_path(cls, name: str=None):
        
        if name == "isbi":
            return os.path.join(cls.root, "ISBI")
        elif name == "pku":
            return os.path.join(cls.root, "PKU")
        elif name == "aariz":
            return os.path.join(cls.root, "Aariz")
        else:
            raise ValueError("\'{}\' dataset doesn't exists in your paths.py file".format(name))
