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


#a) Einzelspalt

l=635*10**-9
I_Dunkel = 0.34*10**-9
x_1, I_1 = np.genfromtxt('scripts/data1.txt', unpack=True)
I_1 = I_1/10**7
x_1 = x_1/1000
def I_function(x,x0,A0,b):
   return A0**2*b**2*l**2/(np.pi*b*np.sin(x-x0))**2*np.sin(np.pi*b*np.sin(x-x0)/l)**2

params,covar = curve_fit(I_function, x_1, I_1,p0=(0.001, 10**9, 0.000075))




I_plot = np.linspace(-18/1000, 18/1000, 1000)
plt.cla()
plt.clf()
plt.plot(I_plot*1000, I_function(I_plot, *params)*10**6,'b-', label='Ausgleichskurve')
plt.plot(x_1*1000, I_1*10**6, 'rx', label='Messwerte')
plt.xlabel(r'$\Delta x/\si{\milli\metre}$')
plt.ylabel(r'$I/\si{\micro\ampere}$')
#plt.xlim(-14, 14)
#plt.ylim(0, 2.3/10)
plt.legend(loc='best')
plt.savefig('content/images/Einzelspalt.pdf')

print('x0: ', params[0],'+/-',np.sqrt(covar[0][0]))
print('A0: ', params[1],'+/-',np.sqrt(covar[1][1]))
print('b: ', params[2],'+/-',np.sqrt(covar[2][2]))

#b)Doppelspalt 1

x_2, I_2 =np.genfromtxt('scripts/data2.txt', unpack=True)
x_2 = x_2/1000
I_2 = I_2/10**6

def I_function2(x,x0,b,s):
    return 4*np.cos(np.pi*s*np.sin(x-x0)/l)**2*l**2/(np.pi*b*np.sin(x-x0))**2*np.sin(np.pi*b*np.sin(x-x0)/l)**2

params2,covar2 = curve_fit(I_function2,x_2,I_2)

I_plot2 = np.linspace(-7/1000,7/1000,1000)
plt.cla()
plt.clf()
plt.plot(I_plot2*1000, I_function2(I_plot2, *params2)*10**6,'b-', label='Ausgleichskurve')
plt.plot(x_2*1000, I_2*10**6, 'rx', label='Messwerte')
plt.xlabel(r'$\Delta x/\si{\milli\metre}$')
plt.ylabel(r'$I/\si{\micro\ampere}$')
#plt.xlim(-14, 14)
#plt.ylim(0, 2.3/10)
plt.legend(loc='best')
plt.savefig('content/images/Doppelspalt1.pdf')


print('x0 von Doppelspalt 1: ', params2[0],'+/-',np.sqrt(covar2[0][0]))
print('b von Doppelspalt 1: ', params2[1],'+/-',np.sqrt(covar2[1][1]))
print('g von Doppelspalt 1: ', params2[2],'+/-',np.sqrt(covar2[2][2]))
