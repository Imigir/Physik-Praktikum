import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
Rerr=1.2154265187
Lerr=0.03/1000
Cerr=0.006/1000000000
C=2.098/1000000000
R=115.486820946
L=10.11/1000
x,y = np.genfromtxt('datad.txt', unpack = True)
y=y/1000000
phi=y*360*x*1000
print(phi)
plt.plot(x, phi, 'rx', label='Messwerte')
plt.xlabel(r'$\nu$ / $10^3$Hz')
plt.ylabel('$\phi$ / °')
plt.legend(loc="best")
plt.savefig('d.pdf')



xs = np.linspace(25,40,2000)
horiz_line_data = np.array([135 for i in range(len(xs))])
plt.plot(xs,horiz_line_data, 'b--', label='135°')
horiz_line_data2 = np.array([45 for i in range(len(xs))])
plt.plot(xs,horiz_line_data2, 'b--', label='45°')
horiz_line_data3 = np.array([90 for i in range(len(xs))])
plt.plot(xs,horiz_line_data3, 'b--', label='90°')
plt.plot(xs,np.arctan((-xs*R*C)/(1-L*C*(xs)**2))*360/(2*math.pi), 'b-', label='Theoriekurve')
plt.plot(x, phi, 'rx', label='Messwerte')
plt.xlabel(r'$\nu$ / $10^3$Hz')
plt.ylabel('$\phi$ / °')
plt.xlim(30,37)
plt.legend(loc="best")
plt.savefig('d2.pdf')
o_res=np.sqrt(1/(L*C)-R**2/(2*L**2))/(2*math.pi)
fehres=np.sqrt((R/(2*L**2*o_res))**2*Rerr**2+(1/(2*L*C**2*o_res))**2*Cerr**2+((L-C*R**2)/(2*C*L**3*o_res))**2*Lerr**2)/(2*math.pi)
wurzel=np.sqrt(1/(L*C)+R**2/(4*L**2))
o1=(R/(2*L)+wurzel)/(2*math.pi)
feh1=np.sqrt((1/(2*L)+R/(4*L**2*wurzel))**2*Rerr**2+(1/(2*L*C**2*wurzel))**2*Cerr**2+(R/(2*L**2)+(1/(C*L**2)+R**2/(2*L**3))/(2*wurzel))**2*Lerr**2)/(2*math.pi)
o2=(-R/(2*L)+wurzel)/(2*math.pi)
feh2=np.sqrt((-1/(2*L)+R/(4*L**2*wurzel))**2*Rerr**2+(1/(2*L*C**2*wurzel))**2*Cerr**2+(R/(2*L**2)-(1/(C*L**2)+R**2/(2*L**3))/(2*wurzel))**2*Lerr**2)/(2*math.pi)
print('nu_res: ',o_res,'+-',fehres)
print('nu 1: ',o1,'+-',feh1)
print('nu 2: ',o2,'+-',feh2)
