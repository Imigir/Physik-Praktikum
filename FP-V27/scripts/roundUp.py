import numpy as np

def roundUp(value,position=1):
	return (np.ceil(value/position)*position)