import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
x,y = np.genfromtxt('dataa.txt', unpack = True)
x=x/1000000
U0=20
R1=48.1
L=10.11/1000

def f(x, A0, m):
    return A0*np.exp(-2*math.pi*m*x)

param, cov = curve_fit(f, x, y)
x_plot = np.linspace(0,400/1000000,10000)
plt.plot(x_plot, f(x_plot, *param), 'b-', label='Ausgleichskurve')
plt.plot(x, y, 'rx', label='Messwerte')
plt.xlabel('t / s')
plt.ylabel('U / V')
plt.legend(loc="best")
plt.savefig('a.pdf')
err = np.sqrt(np.diag(cov))
print('A0 =', param[0], '+-', err[0])
print('m =', param[1], '+-', err[1])

Reff=4*math.pi*param[1]*L
fehR=np.sqrt((4*math.pi*L*err[1])**2+(4*math.pi*param[1]*0.03/1000)**2)
print('R_eff =', Reff,'+-', fehR)

Tex=1/(2*math.pi*param[1])
fehT=err[1]/(2*math.pi*(param[1])**2)
fehTr= fehT/Tex*100
print('T_ex =', Tex,'+-', fehT)
print('relativer Fehler: ', fehTr)
