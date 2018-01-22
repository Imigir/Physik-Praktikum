import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
#x=np.linspace(0,10,1000)
#plt.plot(x,x**2,'b-')
#y=np.arctan((-xs*R*C)/(1-L*C*xs**2))*360/(2*math.pi)
x,y=np.genfromtxt('datac.txt',unpack=True)
U0=9.8
u=y/U0
def f(x,a,b):
    return 1/(np.sqrt((1-4*a**2*x**2)**2+b**2*x**2))
param, cov = curve_fit(f, x, y)
x_plot=np.linspace(25,40,10000)
plt.plot(x,u,'rx',label='Messwerte')
plt.plot(x_plot,f(x_plot,*param),'b-',label='Ausgleichskurve')
plt.xlabel('$\omega / 10^3 Hz$')
plt.ylabel(r'$\frac{U_C}{U_G}$')
plt.xlim(30,37)
plt.ylim(0,17)
plt.legend(loc="best")
plt.savefig('test.pdf')
err= np.sqrt(np.diag(cov))
print(param)
print(err)
