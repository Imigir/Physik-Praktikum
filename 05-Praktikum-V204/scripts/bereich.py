<<<<<<< HEAD
def bereich(x, u, o):
	if(x>=u and x<=o):
		return x
	if(x<u):
		return bereich(o - (u-x), u, o)
	if(x>o):
		return bereich(u + (x-o), u, o)
=======
def bereich(x, u, o):
	if(x>=u and x<=o):
		return x
	if(x<u):
		return bereich(o - (u-x), u, o)
	if(x>o):
		return bereich(u + (x-o), u, o)
>>>>>>> d2b442928cab3209a9290759db8902abb16461ed
