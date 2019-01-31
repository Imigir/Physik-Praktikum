import numpy as np

def test_dim(testlist, dim=0):
    """tests if testlist is a list and how many dimensions it has
    returns -1 if it is no list at all, 0 if list is empty 
    and otherwise the dimensions of it"""
    if isinstance(testlist, list):
        if testlist == []:
            return dim
        dim = dim + 1
        dim = test_dim(testlist[0], dim)
        return dim
  
    else:
        if dim == 0:
            return -1
        return dim
