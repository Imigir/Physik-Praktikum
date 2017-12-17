from table import makeTable
from bereich import bereich
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
import scipy.constants as const

#Für alle#####################################
namex = r'$x/\si{\centi\meter}$'
namey = r'$D(x)/\si{\milli\meter}$'
pi = const.pi
g = const.g

def line (x, a):
	return a * x

#eineitig eingespannt#########################

#RUNDSTAB#####################################
m_StabRund = 121.3 * 0.001
r_StabRund = 1/200
l_StabRund = 0.55-0.05 #Länge des Stabes - eingespannte Länge
m_Gewicht = 522 * 0.001
Ir = pi/4 * r_StabRund**4
F = m_Gewicht * g

print('StabRundEinseitig:')
print('Ir:', Ir)
print('F:', F)

def DurchbiegungEinseitig1(x, a):
	return a*(l_StabRund*x**2-(x**3)/3)

x, y1, y2 = np.genfromtxt('scripts/data1.txt', unpack=True)
x = x/100
yd = (y1-y2)/1000
params, covar = curve_fit(DurchbiegungEinseitig1, x, yd, maxfev=1000)
print('a:', params)
print('Kovarianz:', covar)
t = np.linspace(0, l_StabRund, 500)
plt.cla()
plt.clf()
plt.plot(x*100, yd*1000, 'rx', label='Daten')
plt.plot(t*100, DurchbiegungEinseitig1(t, *params)*1000, 'b-', label='Ausgleichskurve')
plt.xlim(0, 50)
plt.xlabel(namex)
plt.ylabel(namey)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/StabRundEinseitig1')
a = unp.uarray(params[0], np.sqrt(covar[0][0]))
E = F/(2*a*Ir)
print('E1 =', E)

makeTable([x[0:int(len(x)/2)]*100, np.around(yd[0:int(len(yd)/2)]*1000, decimals=2)], r'{'+namex+r'} & {'+namey+r'}', 'tabStabRundEinseitig1', ['S[table-format=2.1]', 'S[table-format=1.2]'], ["%3.1f", "%3.2f"])
makeTable([x[int(len(x)/2):]*100, np.around(yd[int(len(yd)/2):]*1000, decimals=2)], r'{'+namex+r'} & {'+namey+r'}', 'tabStabRundEinseitig2', ['S[table-format=2.1]', 'S[table-format=1.2]'], ["%3.1f", "%3.2f"])


plt.cla()
plt.clf()
plt.plot((l_StabRund*(x)**2-(x)**3/3)*10**3, yd*1000, 'rx', label='Daten')
plt.plot((l_StabRund*(t)**2-(t)**3/3)*10**3, DurchbiegungEinseitig1(t, *params)*1000, 'b-', label='Ausgleichsgerade')
plt.xlim(0, 83)
plt.xlabel(r'$\left(L\cdot x^2-\frac{x^3}{3}\right)/\si{\centi\meter\cubed}$')
plt.ylabel(namey)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/StabRundEinseitig2')


#QUARDRATSTAB#################################
a_StabQuadrat = 1/100
l_StabQuadrat = 0.59-0.05 #Länge des Stabes - eingespannte Länge
m_Gewicht = m_Gewicht + 18.93*0.001
Iq = 1/12 * a_StabQuadrat**4
F = m_Gewicht * g

print('StabQuadratEinseitig:')
print('Iq:', Iq)
print('F:', F)

def DurchbiegungEinseitig2(x, a):
	return a*(0.5*x**2-(x**3)/3)

x, y1, y2 = np.genfromtxt('scripts/data2.txt', unpack=True)
x = x/100
yd = (y1-y2)/1000
params, covar = curve_fit(DurchbiegungEinseitig2, x, yd, maxfev=1000)
print('a:', params)
print('Kovarianz:', covar)
t = np.linspace(0, l_StabQuadrat, 500)
plt.cla()
plt.clf()
plt.plot(x*100, yd*1000, 'rx', label='Daten')
plt.plot(t*100, DurchbiegungEinseitig2(t, *params)*1000, 'b-', label='Ausgleichskurve')
plt.xlim(0, 50)
plt.xlabel(namex)
plt.ylabel(namey)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/StabQuadratEinseitig1')
a = unp.uarray(params[0], np.sqrt(covar[0][0]))
E = F/(2*a*Iq)
print('E2 =', E)


makeTable([x[0:int(len(x)/2)]*100, np.around(yd[0:int(len(yd)/2)]*1000, decimals=2)], r'{'+namex+r'} & {'+namey+r'}', 'tabStabQuadratEinseitig1', ['S[table-format=2.1]', 'S[table-format=1.2]'], ["%3.1f", "%3.2f"])
makeTable([x[int(len(x)/2):]*100, np.around(yd[int(len(yd)/2):]*1000, decimals=2)], r'{'+namex+r'} & {'+namey+r'}', 'tabStabQuadratEinseitig2', ['S[table-format=2.1]', 'S[table-format=1.2]'], ["%3.1f", "%3.2f"])

plt.cla()
plt.clf()
plt.plot((l_StabQuadrat*(x)**2-(x)**3/3)*10**3, yd*1000, 'rx', label='Daten')
plt.plot((l_StabQuadrat*(t)**2-(t)**3/3)*10**3, DurchbiegungEinseitig2(t, *params)*1000, 'b-', label='Ausgleichsgerade')
plt.xlim(0, 90)
plt.xlabel(r'$\left(L\cdot x^2-\frac{x^3}{3}\right)/\si{\centi\meter\cubed}$')
plt.ylabel(namey)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/StabQuadratEinseitig2')

#QUARDRATSTABBEIDSEITIG########################
l_Beidseitig = 0.56
m_Gewicht = (2359.5+2332.6+18.9)*0.001
F = m_Gewicht * g

print('StabQuadratBeidseitig:')
print('Iq:', Iq)
print('F:', F)

def DurchbiegungBeidseitig(x, a):
	links = a*(3*l_Beidseitig**2*x[x<l_Beidseitig/2]-4*(x[x<l_Beidseitig/2]**3))
	rechts = a*(4*x[x>=l_Beidseitig/2]**3 -12* l_Beidseitig* x[x>=l_Beidseitig/2]**2 + 9 * l_Beidseitig**2 * x[x>=l_Beidseitig/2] -l_Beidseitig**3 )
	return np.append(links, rechts)
def DurchbiegungLinks(x, a):
	links = a*(3*l_Beidseitig**2*x-4*(x**3))
	return links
def DurchbiegungRechts(x, a):
	rechts = a*(4*x**3 -12* l_Beidseitig* x**2 + 9 * l_Beidseitig**2 * x -l_Beidseitig**3 )
	return rechts

x, y1, y2 = np.genfromtxt('scripts/data3.txt', unpack=True)
x = x/100
yd = (y1-y2)/1000
params, covar = curve_fit(DurchbiegungBeidseitig, x, yd, maxfev=1000)
print('a:', params)
print('Kovarianz:', covar)
t = np.linspace(0, l_Beidseitig, 500)
plt.cla()
plt.clf()
plt.plot(x*100, yd*1000, 'rx', label='Daten')
plt.plot(t*100, DurchbiegungBeidseitig(t, *params)*1000, 'b-', label='Ausgleichskurve')
plt.xlim(t[0]*100, t[-1]*100)
plt.xlabel(namex)
plt.ylabel(namey)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/StabQuadratBeidseitig1')


makeTable([x[0:int(len(x)/2)]*100, yd[0:int(len(yd)/2)]*1000], r'{'+namex+r'} & {'+namey+r'}', 'tabStabQuadratBeidseitig1', ['S[table-format=2.1]', 'S[table-format=1.2]'], ["%3.1f", "%3.2f"])
makeTable([x[int(len(x)/2):]*100, yd[int(len(yd)/2):]*1000], r'{'+namex+r'} & {'+namey+r'}', 'tabStabQuadratBeidseitig2', ['S[table-format=2.1]', 'S[table-format=1.2]'], ["%3.1f", "%3.2f"])

a = unp.uarray(params[0], np.sqrt(covar[0][0]))
E = F/(48*a*Iq)
print('E3 =', E)
t = np.linspace(-3, l_Beidseitig, 50)
plt.cla()
plt.clf()
plt.plot((3*l_Beidseitig**2*x[x<l_Beidseitig/2]-4*(x[x<l_Beidseitig/2]**3))*10**3, yd[:int(len(yd)/2)]*1000, 'rx', label='Daten')
plt.plot((3*l_Beidseitig**2*t-4*(t**3))*10**3, DurchbiegungLinks(t, *params)*1000, 'b-', label='Ausgleichsgerade')
plt.xlim(30, 190)
plt.ylim(0, 3)
plt.xlabel(r'$\left(3L^2 x-4x^3\right)/\si{\centi\meter\cubed}$')
plt.ylabel(namey)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/StabQuadratBeidseitig2')


plt.cla()
plt.clf()
plt.plot((4*x[x>=l_Beidseitig/2]**3 -12* l_Beidseitig* x[x>=l_Beidseitig/2]**2 + 9 * l_Beidseitig**2 * x[x>=l_Beidseitig/2] -l_Beidseitig**3 )*10**3, yd[int(len(yd)/2):]*1000, 'rx', label='Daten')
plt.plot((4*t**3 -12* l_Beidseitig* t**2 + 9 * l_Beidseitig**2 * t -l_Beidseitig**3 )*10**3, DurchbiegungRechts(t, *params)*1000, 'b-', label='Ausgleichsgerade')
plt.xlim(30, 190)
plt.ylim(0, 3)
plt.xlabel(r'$\left(4 x^3 - 12L  x^2 + 9L^2  x - L^3 \right)/\si{\centi\meter\cubed}$')
plt.ylabel(namey)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/StabQuadratBeidseitig3')










