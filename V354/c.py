import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
x,y = np.genfromtxt('datac.txt', unpack = True)
x=x*1000
U0=9.8
Rerr=1.2154265187
Lerr=0.03/1000
Cerr=0.006/1000000000
C=2.098/1000000000
R=115.486820946
L=10.11/1000
u=y/U0
q1=16.02040816
q2=1/(np.sqrt(1/(L*C))*R*C)
fehq=np.sqrt(Rerr**2*L/(C*R**4)+Cerr**2*L**2/(4*R**2*C**3*L)+Lerr**2/(4*R**2*C*L))
dq=(q2-q1)/q1*100

def f(x,a,b):
    return 1/(np.sqrt((1-4*a**2*x**2)**2+b**2*x**2))
#def f(x, A0, m):
#    return A0*np.exp(-2*math.pi*m*x)

param, cov = curve_fit(f, x, y)
x_plot = np.linspace(10000,52000,100000)
plt.plot(x_plot, f(x_plot, *param), 'b-', label='Ausgleichskurve')

#plt.plot(omega,ausgleich,'b-',label='Theoriekurve')
plt.plot(x, u, 'rx', label='Messwerte')
plt.xlabel(r'$\nu$ / Hz')
plt.ylabel(r'$\frac{U_C}{U}$ / V')
plt.xscale('log')
#plt.yscale('log')
plt.legend(loc="best")
plt.savefig('c.pdf')

err = np.sqrt(np.diag(cov))
print('a =', param[0], '+-', err[0])
print('b =', param[1], '+-', err[1])

x=x/1000
Upm=q1/np.sqrt(2)
xs = np.linspace(25,40,2000)
horiz_line_data = np.array([Upm for i in range(len(xs))])
plt.xscale('linear')
#omega1=np.linspace(20,40,100000)
#plt.plot(omega1,ausgleich,'b-',label='Theoriekurve')
x_plot = np.linspace(30,37,10000)
plt.plot(x_plot, f(x_plot, *param), 'b-', label='Ausgleichskurve')
plt.plot(x, u, 'rx', label='Messwerte')
plt.plot(xs,horiz_line_data, 'b--', label='')
plt.xlabel(r'$\nu$ /$10^3$ Hz')
plt.ylabel(r'$\frac{U_C}{U}$ / V')
plt.xlim(30,37)
plt.ylim(0,17)
plt.savefig('c2.pdf')
b=R/L
fehb=np.sqrt(Rerr**2/L**2+R**2*Lerr**2/L**4)
#err = np.sqrt(np.diag(cov))
#print('A0 =', param[0], '+-', err[0])
#print('m =', param[1], '+-', err[1])
print('nu: ',x)
print('U_C: ',y)
print('U_C/U: ',u)
print('q_ex: ',q1)
print('q_the: ',q2,'+-',fehq)
print('Abweichung: ',dq)
print('theoretische Breite der Resonanzkurve: ',b,'+-',fehb)
