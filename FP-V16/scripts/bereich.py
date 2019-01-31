def bereich(x, u, o):
	if(x>=u and x<=o):
		return x
	if(x<u):
		return bereich(o - (u-x), u, o)
	if(x>o):
		return bereich(u + (x-o), u, o)
