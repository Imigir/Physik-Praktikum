<<<<<<< HEAD
﻿import numpy as np
from scipy import stats
def weighted_avg_and_sem(values, weights):
	average = np.average(values, weights=weights)
	variance = np.average((values-average)**2, weights=weights)/(len(values)-1)
	return (average, np.sqrt(variance))

def avg_and_sem(values):
	average = np.average(values)
	variance = np.average((values-average)**2)/(len(values)-1)
=======
﻿import numpy as np
from scipy import stats
def weighted_avg_and_sem(values, weights):
	average = np.average(values, weights=weights)
	variance = np.average((values-average)**2, weights=weights)/(len(values)-1)
	return (average, np.sqrt(variance))

def avg_and_sem(values):
	average = np.average(values)
	variance = np.average((values-average)**2)/(len(values)-1)
>>>>>>> d2b442928cab3209a9290759db8902abb16461ed
	return (average, np.sqrt(variance))