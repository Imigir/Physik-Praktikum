import matplotlib.pyplot as plt
import numpy as np
import math

x, y = np.genfromtxt('datad.txt', unpack='true')
x *= 1e3
y /= 1e6
Rerr=1.2154265187
Lerr=0.03/1000
Cerr=0.006/1000000000
C=2.098/1000000000
R=115.486820946
L=10.11/1000
phi1=y*360*x

t = np.linspace(1e-10, 100, 10000)*10**3
phi = np.arctan(((2*np.pi*t)**2*L*C-1)/(2*np.pi*t*R*C))*180/np.pi+90

yposition = [45, 135]
for yc in yposition:
    plt.axhline(y=yc, color='xkcd:grey', linestyle='--')

xposition = [31.75*10**3, 42.5*10**3]
for xc in xposition:
    plt.axvline(x=xc, color='xkcd:grey', linestyle='--')

plt.axhline(y=76.75, color='g', linestyle='--')
plt.axvline(x=35.5*10**3, color='g', linestyle='--')

plt.plot(35.5*10**3, 76.75, 'k.')
plt.plot(31.75*10**3, 45, 'k.')
plt.plot(42.5*10**3, 135, 'k.')
plt.text(36*10**3, 69.75, r'$\nu_\mathrm{res}$', fontsize=12)
plt.text(32.25*10**3, 38, r'$\nu_2$', fontsize=12)
plt.text(43*10**3, 128, r'$\nu_1$', fontsize=12)

plt.plot(t, phi, 'b-', label='Theoriekurve')

plt.plot(x, phi1, 'rx', label='Messwerte')
plt.xlabel(r'$\nu \,/\, \mathrm{Hz}$')
plt.ylabel(r'$\Phi \,/\, \mathrm{°}$')
plt.xlim(30*10**3, 37*10**3)
plt.ylim(0, 180)

plt.grid()
plt.legend()
# plt.show()
plt.savefig('phase.pdf')

print('Experimentelle Werte')
print('ny_res =', 35.5*10**3)
print('ny_1 =', 42.5*10**3)
print('ny_2 =', 31.75*10**3)

print('Theoriewerte')

a = -1/(4*np.pi*L*C**2*np.sqrt(1/(L*C)-R**2)/(2*L**2))
b = -R/(4*np.pi*L**2*np.sqrt(1/(L*C)-R**2/(2*L**2)))
c = 1/(4*np.pi*np.sqrt(1/(L*C)-R**2/(2*L**2)))*(-1/(L**2*C)+2*R**2/(2*L**3))

nyres = np.sqrt(1/(L*C)-R**2/(2*L**2))/(2*np.pi)
errnyres=np.sqrt((R/(2*L**2*nyres))**2*Rerr**2+(1/(2*L*C**2*nyres))**2*Cerr**2+((L-C*R**2)/(2*C*L**3*nyres))**2*Lerr**2)/(2*math.pi)

print('ny_res =', nyres, '+-', errnyres)

ny1 = 1/(2*np.pi)*(R/(2*L)+np.sqrt((R**2/(4*L**2))+(1/(L*C))))
d = -1/(4*np.pi*L*C**2*np.sqrt((R**2/(4*L**2)+1/(L*C))))
e = 1/(2*L)*(1/(2*L)+R/(4*L**2*np.sqrt(R**2/(4*L**2)+1/(L*C))))
f = 1/(2*np.pi)*(-R/(2*L**2)+1/(2*np.sqrt(R**2/(4*L**2)+1/(L*C)))*((-2*R**2)/(4*L**3)-1/(L**2*C)))
wurzel=np.sqrt(1/(L*C)+R**2/(4*L**2))
errny1 = feh1=np.sqrt((1/(2*L)+R/(4*L**2*wurzel))**2*Rerr**2+(1/(2*L*C**2*wurzel))**2*Cerr**2+(R/(2*L**2)+(1/(C*L**2)+R**2/(2*L**3))/(2*wurzel))**2*Lerr**2)/(2*math.pi)


print('ny_1 =', ny1, '+-', errny1)

ny2 = 1/(2*np.pi)*(-R/(2*L)+np.sqrt((R**2/(4*L**2))+(1/(L*C))))
g = 1/(2*L)*(-1/(2*L)+R/(4*L**2*np.sqrt(R**2/(4*L**2)+1/(L*C))))
h = 1/(2*np.pi)*(R/(2*L**2)+1/(2*np.sqrt(R**2/(4*L**2)+1/(L*C)))*((-2*R**2)/(4*L**3)-1/(L**2*C)))

errny2=np.sqrt((-1/(2*L)+R/(4*L**2*wurzel))**2*Rerr**2+(1/(2*L*C**2*wurzel))**2*Cerr**2+(R/(2*L**2)-(1/(C*L**2)+R**2/(2*L**3))/(2*wurzel))**2*Lerr**2)/(2*math.pi)


print('ny_2 =', ny2, '+-', errny2)

print('Abweichung')

xx = np.abs((nyres-35.5*10**3)/nyres)
print('... von ny_res =', xx)

yy = np.abs((ny1-42.5*10**3)/ny1)
print('... von ny_1 =', yy)

zz = np.abs((ny2-31.75*10**3)/ny2)
print('... von ny_2 =', zz)
