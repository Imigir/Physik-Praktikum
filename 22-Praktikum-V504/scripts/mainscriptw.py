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

#a)
"""
print('a)')
"""
UB_21, I_21 = np.genfromtxt(r'scripts/data1.txt', unpack=True)
UB_21 = UB_21*5
I_21 = I_21/1000
IH_21 = 2.1
UH_21 = 4.75

makeTable([UB_21,I_21*10**6], r'{'+r'$U_\text{B}/\si{\volt}$'+r'} & {'+r'$I_\text{2,1}/\si{\micro\ampere}$'+r'}' ,'tab21' , ['S[table-format=3.0]' , 'S[table-format=3.0]' ] , ["%3.0f", "%3.0f"])

"""
plt.cla()
plt.clf()
plt.plot(UB_21, I_21*10**6, 'rx', label=r'$I_\text{H} = \SI{2,1}{\ampere}$')
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$I/\si{\micro\ampere}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/IH_21.pdf')

print('2,1 finished')
"""
UB_22, I_22 = np.genfromtxt(r'scripts/data2.txt', unpack=True)
UB_22 = UB_22*5
I_22 = I_22/1000
IH_22 = 2.2
UH_22 = 5

makeTable([UB_22,I_22*10**6], r'{'+r'$U_\text{B}/\si{\volt}$'+r'} & {'+r'$I_\text{2,2}/\si{\micro\ampere}$'+r'}' ,'tab22' , ['S[table-format=3.0]' , 'S[table-format=3.0]' ] , ["%3.0f", "%3.0f"])

"""
plt.cla()
plt.clf()
plt.plot(UB_22, I_22*10**6, 'rx', label=r'$I_\text{H} = \SI{2,2}{\ampere}$')
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$I/\si{\micro\ampere}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/IH_22.pdf')

print('2,2 finished')
"""
"""
plt.cla()
plt.clf()
plt.plot(UB_21, I_21*10**6, 'rx', label=r'$I_\text{H} = \SI{2,1}{\ampere}$')
plt.plot(UB_22, I_22*10**6, 'gx', label=r'$I_\text{H} = \SI{2,2}{\ampere}$')
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$I/\si{\micro\ampere}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/IH_21-2.pdf')

print('2,1-2 finished')
"""
UB_23, I_23 = np.genfromtxt(r'scripts/data3.txt', unpack=True)
UB_23 = UB_23*5
I_23 = I_23/1000
IH_23 = 2.3
UH_23 = 5.5

makeTable([UB_23,I_23*10**6], r'{'+r'$U_\text{B}/\si{\volt}$'+r'} & {'+r'$I_\text{2,3}/\si{\micro\ampere}$'+r'}' ,'tab23' , ['S[table-format=3.0]' , 'S[table-format=3.0]' ] , ["%3.0f", "%3.0f"])

makeTable([UB_23,I_21*10**6,I_22*10**6,I_23*10**6], r'{'+r'$U_\text{B}/\si{\volt}$'+r'} & {'+r'$I_\text{2,1}/\si{\micro\ampere}$'+r'} & {'+r'$I_\text{2,2}/\si{\micro\ampere}$'+r'} & {'+r'$I_\text{2,3}/\si{\micro\ampere}$'+r'}' ,'tab21-3' , ['S[table-format=3.0]' , 'S[table-format=3.0]',  'S[table-format=3.0]',  'S[table-format=3.0]' ] , ["%3.0f", "%3.0f", "%3.0f", "%3.0f"])

"""
plt.cla()
plt.clf()
plt.plot(UB_23, I_23*10**6, 'rx', label=r'$I_\text{H} = \SI{2,3}{\ampere}$')
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$I/\si{\micro\ampere}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/IH_23.pdf')

print('2,3 finished')
"""
"""
plt.cla()
plt.clf()
plt.plot(UB_21, I_21*10**6, 'rx', label=r'$I_\text{H} = \SI{2,1}{\ampere}$')
plt.plot(UB_22, I_22*10**6, 'gx', label=r'$I_\text{H} = \SI{2,2}{\ampere}$')
plt.plot(UB_23, I_23*10**6, 'bx', label=r'$I_\text{H} = \SI{2,3}{\ampere}$')
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$I/\si{\micro\ampere}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/IH_21-3.pdf')

print('2,1-3 finished')
"""
UB_24, I_24 = np.genfromtxt(r'scripts/data4.txt', unpack=True)
UB_24 = UB_24*10
I_24 = I_24/1000
IH_24 = 2.4
UH_24 = 5.95

makeTable([UB_24,I_24*10**6], r'{'+r'$U_\text{B}/\si{\volt}$'+r'} & {'+r'$I_\text{2,4}/\si{\micro\ampere}$'+r'}' ,'tab24' , ['S[table-format=3.0]' , 'S[table-format=4.0]' ] , ["%3.0f", "%4.0f"])

"""
plt.cla()
plt.clf()
plt.plot(UB_24, I_24*10**6, 'rx', label=r'$I_\text{H} = \SI{2,4}{\ampere}$')
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$I/\si{\micro\ampere}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/IH_24.pdf')

print('2,4 finished')
"""
UB_25, I_25 = np.genfromtxt(r'scripts/data5.txt', unpack=True)
UB_25 = UB_25*10
I_25 = I_25/1000
IH_25 = 2.5
UH_25 = 6

makeTable([UB_25,I_25*10**6], r'{'+r'$U_\text{B}/\si{\volt}$'+r'} & {'+r'$I_\text{2,5}/\si{\micro\ampere}$'+r'}' ,'tab25' , ['S[table-format=3.0]' , 'S[table-format=4.0]' ] , ["%3.0f", "%4.0f"])

makeTable([UB_25,I_24*10**6,I_25*10**6], r'{'+r'$U_\text{B}/\si{\volt}$'+r'} & {'+r'$I_\text{2,4}/\si{\micro\ampere}$'+r'} & {'+r'$I_\text{2,5}/\si{\micro\ampere}$'+r'}' ,'tab24-5' , ['S[table-format=3.0]', 'S[table-format=4.0]' , 'S[table-format=4.0]' ] , ["%3.0f", "%4.0f", "%4.0f"])

"""
plt.cla()
plt.clf()
plt.plot(UB_25, I_25*10**6, 'rx', label=r'$I_\text{H} = \SI{2,5}{\ampere}$')
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$I/\si{\micro\ampere}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/IH_25.pdf')

print('2,5 finished')
"""
"""
plt.cla()
plt.clf()
plt.plot(UB_21, I_21*10**6, 'rx', label=r'$I_\text{H} = \SI{2,1}{\ampere}$')
plt.plot(UB_22, I_22*10**6, 'gx', label=r'$I_\text{H} = \SI{2,2}{\ampere}$')
plt.plot(UB_23, I_23*10**6, 'bx', label=r'$I_\text{H} = \SI{2,3}{\ampere}$')
plt.plot(UB_24, I_24*10**6, 'yx', label=r'$I_\text{H} = \SI{2,4}{\ampere}$')
plt.plot(UB_25, I_25*10**6, 'kx', label=r'$I_\text{H} = \SI{2,5}{\ampere}$')
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$I/\si{\micro\ampere}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/IH_21-5.pdf')

print('2,1-5 finished')
"""
#b)
print('b)')
UB_log = np.log(UB_25)
I_log = np.log(I_25)

paramsLinear, errorsLinear, sigma_y = linregress(UB_log,I_log)
steigung = unp.uarray(paramsLinear[0], errorsLinear[0])
achsenAbschnitt = unp.uarray(paramsLinear[1], errorsLinear[1])

print('steigung:', steigung)
print('Achsenabschnitt:', achsenAbschnitt)
"""
plt.cla()
plt.clf()
x_plot = np.linspace(2,6)
plt.plot(UB_log, I_log, 'rx', label='Messwerte')
plt.plot(x_plot, x_plot*paramsLinear[0]+paramsLinear[1], 'b-', label='Ausgleichsgerade')
plt.xlim(2,6)
plt.xlabel(r'$\log(U/\si{\volt})$')
plt.ylabel(r'$\log(I/\si{\ampere})$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/IH_25_log.pdf')
"""
#c)
print('c)')
UB_25, I_25 = np.genfromtxt(r'scripts/data6.txt', unpack=True)
I_25 = I_25*10**(-9)
UB_25 = UB_25*(-0.1)
UB_25 = UB_25+10**6*I_25
I_log = np.log(I_25)

makeTable([UB_25,I_25*10**9], r'{'+r'$U_\text{B}/\si{\volt}$'+r'} & {'+r'$I_\text{Anlauf}/\si{\nano\ampere}$'+r'}' ,'tabAnlaufstrom' , ['S[table-format=0.2]' , 'S[table-format=2.1]' ] , ["%0.2f", "%2.1f"])

e0 = const.value("electron volt")
kB = const.value("Boltzmann constant")

paramsLinear, errorsLinear, sigma_y = linregress(UB_25,I_log)
steigung = unp.uarray(paramsLinear[0], errorsLinear[0])
achsenAbschnitt = unp.uarray(paramsLinear[1], errorsLinear[1])
T = e0/(kB*steigung)

print('steigung:', steigung)
print('Achsenabschnitt:', achsenAbschnitt)
print('T=', T)
"""
plt.cla()
plt.clf()
x_plot = np.linspace(-1,0.2)
plt.plot(UB_25, I_log, 'rx', label='Messwerte')
plt.plot(x_plot, x_plot*paramsLinear[0]+paramsLinear[1], 'b-', label='Ausgleichsgerade')
plt.xlim(-1,0.2)
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$\log(I/\si{\ampere})$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/Anlaufstrom.pdf')
"""
#d)
print('d)')
UH = np.array([UH_21,UH_22,UH_23,UH_24,UH_25])
IH = np.array([IH_21,IH_22,IH_23,IH_24,IH_25])
WH = UH*IH
A = 0.32/10**4
eta = 0.28
sigma = 5.7*10**(-8)

TS = ((WH-1)/(A*eta*sigma))**(0.25)

print('TS=', TS)

#e)
print('e)')
h = const.value("Planck constant")
m0 = const.value("electron mass")
IS = np.array([230, 600, 1500, 3500, 6000])*10**(-6)

phi = -np.log(IS*h**3/(4*np.pi*e0*m0*A*kB**2*TS**2))*kB*TS/e0
phi_m = unp.uarray(avg_and_sem(phi)[0],avg_and_sem(phi)[1])

print('phi=', phi)
print('phi_m=', phi_m)

#weitereTabelle

makeTable([IH,UH,WH,IS*10**6,TS,phi], r'{'+r'$I_\text{H}/\si{\ampere}$'+r'} & {'+r'$U_\text{H}/\si{\volt}$'+r'} & {'+r'$W_\text{H}/\si{\watt}$'+r'} & {'+r'$I_\text{S}/\si{\micro\ampere}$'+r'} & {'+r'$T_\text{S}/\si{\kelvin}$'+r'} & {'+r'$\phi/\si{\electronvolt}$'+r'}' ,'tabSTA' , ['S[table-format=1.1]', 'S[table-format=1.2]' , 'S[table-format=2.1]', 'S[table-format=4.0]', 'S[table-format=3.0]', 'S[table-format=1.2]'] ,  ["%1.1f", "%1.2f", "%2.1f", "%4.0f", "%3.0f", "%1.2f"])

