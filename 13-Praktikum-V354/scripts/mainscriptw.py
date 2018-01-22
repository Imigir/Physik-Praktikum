from table import makeTable
from table import makeNewTable
from linregress import linregress
from customFormatting import *
from bereich import bereich
from weightedavgandsem import weighted_avg_and_sem
from weightedavgandsem import avg_and_sem
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
import scipy.constants as const

# BackwardsVNominal = []
# BackwardsVStd = []
# for value in BackwardsV:
#     BackwardsVNominal.append(unp.nominal_values(value))
#     BackwardsVStd.append(unp.std_devs(value))
# BackwardsVNominal = np.array(BackwardsVNominal)
# BackwardsVStd = np.array(BackwardsVStd)

# einfacher:
# BackwardsVNominal = unp.nominal_values(BackwardsV)
# BackwardsVStd = unp.std_devs(BackwardsV)

# makeTable([Gaenge, ForwardsVNominal, ForwardsVStd, ], r'{Gang} & \multicolumn{2}{c}{$v_\text{v}/\si[per-mode=reciprocal]{\centi\meter\per\second}$} & ', 'name', ['S[table-format=2.0]', 'S[table-format=2.3]', ' @{${}\pm{}$} S[table-format=1.3]', ], ["%2.0f", "%2.3f", "%2.3f",])

#[per-mode=reciprocal],[table-format=2.3,table-figures-uncertainty=1]

# unp.uarray(np.mean(), stats.sem())
# unp.uarray(*avg_and_sem(values)))
# unp.uarray(*weighted_avg_and_sem(unp.nominal_values(bneuDiff), 1/unp.std_devs(bneuDiff)))

# plt.cla()
# plt.clf()
# plt.plot(ForwardsVNominal*100, DeltaVForwardsNominal, 'gx', label='Daten mit Bewegungsrichtung aufs Mikrofon zu')
# plt.plot(BackwardsVNominal*100, DeltaVBackwardsNominal, 'rx', label='Daten mit Bewegungsrichtung vom Mikrofon weg')
# plt.ylim(0, line(t[-1], *params)+0.1)
# plt.xlim(0, t[-1]*100)
# plt.xlabel(r'$v/\si{\centi\meter\per\second}$')
# plt.ylabel(r'$\Delta f / \si{\hertz}$')
# plt.legend(loc='best')
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/'+'VgegenDeltaV')

# a = unp.uarray(params[0], np.sqrt(covar[0][0]))
# params = unp.uarray(params, np.sqrt(np.diag(covar)))
# makeNewTable([convert((r'$c_\text{1}$',r'$c_\text{2}$',r'$T_{\text{A}1}$',r'$T_{\text{A}2}$',r'$\alpha$',r'$D_1$',r'$D_2$',r'$A_1$',r'$A_2$',r'$A_3$',r'$A_4$'),strFormat),convert(np.array([paramsGes2[0],paramsGes1[0],deltat2*10**6,deltat1*10**6,-paramsDaempfung[0]*2,4.48*10**-6 *paramsGes1[0]/2*10**3, 7.26*10**-6 *paramsGes1[0]/2*10**3, (VierteMessung-2*deltat2*10**6)[0]*10**-6 *1410 /2*10**3, unp.uarray((VierteMessung[1]-VierteMessung[0])*10**-6 *1410 /2*10**3, 0), unp.uarray((VierteMessung[2]-VierteMessung[1])*10**-6 *2500 /2*10**3, 0),unp.uarray((VierteMessung[3]-VierteMessung[2])*10**-6 *1410 /2*10**3, 0)]),unpFormat,[[r'\meter\per\second',"",True],[r'\meter\per\second',"",True],[r'\micro\second',"",True],[r'\micro\second',"",True],[r'\per\meter',"",True],[r'\milli\meter',"",True],[r'\milli\meter',"",True],[r'\milli\meter',"",True],[r'\milli\meter',r'1.3f',True],[r'\milli\meter',r'1.3f',True],[r'\milli\meter',r'2.2f',True]]),convert(np.array([2730,2730]),floatFormat,[r'\meter\per\second','1.0f',True])+convert((r'-',r'-'),strFormat)+convert(unp.uarray([57,6.05,9.9],[2.5,0,0]),unpFormat,[[r'\per\meter',"",True],[r'\milli\meter',r'1.2f',True],[r'\milli\meter',r'1.2f',True]])+convert((r'-',r'-',r'-',r'-'),strFormat),convert(np.array([(2730-paramsGes2[0])/2730*100,(2730-paramsGes1[0])/2730*100]),unpFormat,[r'\percent','',True])+convert((r'-',r'-'),strFormat)+convert(np.array([(-paramsDaempfung[0]*2-unp.uarray(57,2.5))/unp.uarray(57,2.5)*100,(4.48*10**-6 *paramsGes1[0]/2*10**3-6.05)/6.05*100, (-7.26*10**-6 *paramsGes1[0]/2*10**3+9.90)/9.90*100]),unpFormat,[r'\percent','',True])+convert((r'-',r'-',r'-',r'-'),strFormat)],r'{Wert}&{gemessen}&{Literaturwert\cite{cAcryl},\cite{alphaAcryl}}&{Abweichung}','Ergebnisse', ['c ','c',r'c','c'])

L=unp.uarray(10.11,0.03)/1000
C=unp.uarray(2.098,0.006)/10**9
R=unp.uarray(115.486820946,1.2154265187)


#a
def f(x, A0, m):
    return A0*np.exp(-2*np.pi*m*x)

print('a)')
x,y = np.genfromtxt('scripts/data1.txt', unpack = True)
x=x/10**6
U0=20
	
params, covar = curve_fit(f, x, y)
x_plot = np.linspace(-20/10**6,420/10**6)
plt.plot(x, y, 'rx', label='Messwerte')
plt.plot(x_plot, f(x_plot, *params), 'b-', label='Ausgleichskurve')
plt.xlabel('t / s')
plt.ylabel('U / V')
plt.xlim(-20/10**6,420/10**6)
plt.legend(loc="best")
plt.savefig('content/images/Grapha.pdf')

A0 = unp.uarray(params[0], np.sqrt(covar[0][0]))
m = unp.uarray(params[1], np.sqrt(covar[1][1]))
print('A0 =', A0)
print('m =', m)

Reff=4*np.pi*m*L
print('R_eff =', Reff)

tau=1/(2*np.pi*m)
print('tau =', tau)


#c
def AcT(f, a, b):
	return 1/np.sqrt((1-(2*np.pi*f*a)**2)**2+(b*2*np.pi*f)**2)

print('c)')
f, Ac = np.genfromtxt('scripts/data2.txt', unpack=True)
f = f*1000
f2 = f/10**3
A =  9.8
Ar = Ac/A

params, covar = curve_fit(AcT, f2, Ar)

t = np.linspace(f[0]*0.98, f[-1]*1.02, 10**3)
t2 = t / 10**3
plt.cla()
plt.clf()
plt.plot(f, Ar, 'rx', label='Messwerte')
plt.plot(t, AcT(t2, *params), 'b-', label='Ausgleichskurve')
plt.xscale('log')
plt.xlim(f[0]*0.98, f[-1]*1.02)
plt.xlabel(r'$\nu/\si{\hertz}$')
plt.ylabel(r'$\frac{U_C}{U}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/Graphc1')

LC = (unp.uarray(params[0], np.sqrt(covar[0][0]))/10**3)**2
RC = unp.uarray(-params[1], np.sqrt(covar[1][1]))/10**3

print('LC = ', LC)
print('RC = ', RC)
print('LC errechnet = ', L*C)
print('RC errechnet = ', R*C)

q = unp.sqrt(LC)/RC
print('Güte q = ', q)
qer = unp.sqrt(L*C)/(R*C)
print('Güte q errechnet = ', qer)
print('Abweichung in Prozent = ', (qer-q)/qer * 100)


a = (RC**2-2*LC)/(2*LC**2)
print('a = ', a)
w1 = unp.sqrt(-a - unp.sqrt(a**2 - (1-2/q**2)/LC**2))
w2 = unp.sqrt(-a + unp.sqrt(a**2 - (1-2/q**2)/LC**2))
print('w- = ', w1)
print('w+ = ', w2)
print('f- = ', w1 / (2*np.pi))
print('f+ = ', w2 / (2*np.pi))
print('Breite der Ressonanzkurve = ', (w2 - w1) / (2*np.pi))

aer = ((R*C)**2-2*L*C)/(2*(L*C)**2)
print('a errechnet = ', aer)
w1er = unp.sqrt(-aer - unp.sqrt(aer**2 - (1-2/qer**2)/(L*C)**2))
w2er = unp.sqrt(-aer + unp.sqrt(aer**2 - (1-2/qer**2)/(L*C)**2))
print('w- errechnet = ', w1er)
print('w+ errechnet = ', w2er)
print('f- errechnet = ', w1er / (2*np.pi))
print('f+ errechnet = ', w2er / (2*np.pi))
print('Breite der Ressonanzkurve errechnet = ', (w2er - w1er) / (2*np.pi))
print('Abweichung in Prozent = ',((w2 - w1) / (2*np.pi)-(w2er - w1er) / (2*np.pi))/((w2er - w1er) / (2*np.pi))*100)


f2 = f[4:-4]
Ar2 = Ar[4:-4]

t = np.linspace(f2[0]*0.98, f2[-1]*1.02, 10**5)
t2 = t / 10**3
plt.cla()
plt.clf()
plt.plot(f2, Ar2, 'rx', label='Daten')
plt.plot(t, AcT(t2, *params), 'b-', label='Fit')
plt.xlim(f2[0]*0.98, f2[-1]*1.02)
plt.xlabel(r'$f_{\text{Antrieb}}/\si{\hertz}$')
plt.ylabel(r'$A_C/A_{\text{Antrieb}}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/Graphc2')