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
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
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

N,U,I = np.genfromtxt('scripts/data1.txt', unpack = True)
N_err = np.sqrt(N)
N = N/60
N_err = N_err/60
N = unp.uarray(N, N_err)
I = I/10**6

#a)
paramsLinear, errorsLinear, R = linregress(U[3:37], noms(N[3:37]))

steigung = unp.uarray(paramsLinear[0], errorsLinear[0])
achsenAbschnitt = unp.uarray(paramsLinear[1], errorsLinear[1])

N_min = steigung*U[3]+achsenAbschnitt
print('N_min =', N_min)
print('Steigung des Plateaus =', steigung)
print('Steigung des Plateaus in %/100V =', steigung/N_min*100*100)
print('Achsenabschnitt des Plateaus =', achsenAbschnitt)
print('Länge des Plateaus =', U[36]-U[3])

plt.cla()
plt.clf()
x_plot = np.linspace(200,800)
plt.errorbar(U, noms(N), yerr=stds(N), fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot, x_plot*paramsLinear[0]+paramsLinear[1], 'b-', linewidth=0.8, label='Ausgleichsgerade des Plateaus')
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$N/\si{\becquerel}$')
plt.xlim(280,720)
plt.legend(loc="best")
plt.savefig('content/images/Graph1.1.pdf')

plt.cla()
plt.clf()
x_plot = np.linspace(200,800)
plt.errorbar(U[3:37], noms(N[3:37]), yerr=stds(N[3:37]), fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot, x_plot*paramsLinear[0]+paramsLinear[1], 'b-', linewidth=0.8, label='Ausgleichsgerade des Plateaus')
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$N/\si{\becquerel}$')
plt.xlim(320,670)
plt.legend(loc="best")
plt.savefig('content/images/Graph1.2.pdf')

#c)
N1w = 17131
N1 = unp.uarray(N1w/60, np.sqrt(N1w)/60)
N2w = 1063
N2 = unp.uarray(N2w/60, np.sqrt(N2w)/60)
N12w = 18124
N12 = unp.uarray(N12w/60, np.sqrt(N12w)/60)

T1 = unp.uarray(100*10**(-6),25*10**(-6))
T2 = (N1+N2-N12)/(2*N1*N2)

DT=(T2-T1)/T1

print('T1 =', T1)
print('T2 =', T2)
print('Abweichung T1 zu T2 in % =', DT*100)

#d)
e = const.e
N[0] = 1
DQ = I/N
DQe = DQ/e
N[0] = 0

paramsLinear, errorsLinear, R = linregress(U, noms(DQe))

steigung = unp.uarray(paramsLinear[0], errorsLinear[0])
achsenAbschnitt = unp.uarray(paramsLinear[1], errorsLinear[1])

print('Steigung =', steigung)
print('Achsenabschnitt =', achsenAbschnitt)
print('Bestimmtheitsmaß =', R)

plt.cla()
plt.clf()
x_plot = np.linspace(200,800)
plt.errorbar(U, noms(DQe), yerr=stds(DQe), fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
#plt.plot(U, noms(DQe), 'rx',label='Berechnete Werte')
plt.plot(x_plot, x_plot*paramsLinear[0]+paramsLinear[1], 'b-', label='Ausgleichsgerade')
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$\Delta Q/e$')
plt.xlim(280,720)
plt.legend(loc="best")
plt.savefig('content/images/Graph2.pdf')

makeTable([noms(N)*60, stds(N)*60, noms(N), stds(N), U, I*10**6, noms(DQe)*10**(-8), stds(DQe)*10**(-8)], r'\multicolumn{2}{c}{'+r'$N_.{minute}/\si{1\per\minute}$'+r'} & \multicolumn{2}{c}{'+r'$N/\si{\becquerel}$'+r'} & {'+r'$U/\si{\volt}$'+r'} & {'+r'$I/10^{-6}\si{\ampere}$'+r'} & \multicolumn{2}{c}{'+r'$\Delta Q/10^{8}\mathrm{e}$'+r'}', 'tab1', ['S[table-format=5.0]', ' @{${}\pm{}$} S[table-format=3.0]', 'S[table-format=3.0]', ' @{${}\pm{}$} S[table-format=1.0]', 'S[table-format=3.0]', 'S[table-format=1.2]', 'S[table-format=3.1]', ' @{${}\pm{}$} S[table-format=1.1]'], ["%5.0f", "%3.0f", "%3.0f", "%1.0f", "%3.0f", "%1.2f", "%3.1f", "%1.1f"])
