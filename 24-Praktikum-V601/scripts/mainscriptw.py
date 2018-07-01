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

#mittlere Weglängen
a = 1/100
T0 = 273.15
T = np.array([23.5+T0, 144+T0, 174.2+T0, 108+T0])
w = 0.0029/(5.5*10**7)*np.exp(6867/T)/100	
aw = a/w

print('T = ', T)
print('w = ', w)
print('a/w = ', aw)

#Skalierungsfaktoren
da1, da2, db, dc = np.genfromtxt(r'scripts/data1.txt', unpack=True)
da1 = da1/1000
da2 = da2/1000
db = db/5000
dc = dc/5000

da1m = avg_and_sem(da1[0:10])
da2m = avg_and_sem(da2[0:10])
dbm = avg_and_sem(db)
dcm = avg_and_sem(dc[0:10])

fa1 = 1/da1m[0]
fa2 = 1/da2m[0]
fb = 1/dbm[0]
fc = 1/dcm[0]

da1m = unp.uarray(da1m[0], da1m[1])
da2m = unp.uarray(da2m[0], da2m[1])
dbm = unp.uarray(dbm[0], dbm[1])
dcm = unp.uarray(dcm[0], dcm[1])

print('da1 = ', da1m)
print('da2 = ', da2m)
print('db = ', dbm)
print('dc = ', dcm)

makeTable([da1*1000, da2*1000, db*1000, dc*1000], r'{'+r'$d_\text{a1}/(\si{\milli\metre\per\volt})$'+r'} & {'+r'$d_\text{a2}/(\si{\milli\metre\per\volt})$'+r'} & {'+r'$d_\text{b}/(\si{\milli\metre\per\volt})$'+r'} & {'+r'$d_\text{c}/(\si{\milli\metre\per\volt})$'+r'}' ,'tabAbstaende' , ['S[table-format=2.0]', 'S[table-format=2.0]', 'S[table-format=1.0]', 'S[table-format=1.0]'] , ["%2.0f", "%2.0f", "%1.0f", "%1.0f"])

print('fa1 = ', 1/da1m)
print('fa2 = ', 1/da2m)
print('fb = ', 1/dbm)
print('fc = ', 1/dcm)

#a)
#1)
print('a)')
U1, UI1 = np.genfromtxt(r'scripts/data2.txt', unpack=True)
U1 = U1/1000*fa1
"""
plt.cla()
plt.clf()
plt.plot(U1, UI1, 'rx', label='Messwerte')
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$I/U$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/fig1.pdf')
"""
k1 = 11-U1[15]/fa1/da1m
print('k1 = ', k1)

#2)
U2, UI2 = np.genfromtxt(r'scripts/data3.txt', unpack=True)
U2 = U2/1000*fa2
"""
plt.cla()
plt.clf()
plt.plot(U2, UI2, 'rx', label='Messwerte')
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$I/U$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/fig2.pdf')
"""
makeTable([U1, UI1, U2, UI2], r'{'+r'$U_.{A1}/\si{\volt}$'+r'} & {'+r'$I_1/U_.{A1}$'+r'} & {'+r'$U_.{A2}/\si{\volt}$'+r'} & {'+r'$I_2/U_.{A2}$'+r'}' ,'tabEnergieverteilung' , ['S[table-format=1.2]', 'S[table-format=2.2]', 'S[table-format=1.2]', 'S[table-format=1.2]'] , ["%1.2f", "%2.2f", "%1.2f", "%1.2f"])

#b)
print('b)')
n, a = np.genfromtxt(r'scripts/data4.txt', unpack=True)
a = a/1000*fb

paramsLinear, errorsLinear, sigma_y = linregress(n,a)
steigung = unp.uarray(paramsLinear[0], errorsLinear[0])
achsenAbschnitt = unp.uarray(paramsLinear[1], errorsLinear[1])

#k2 = a[0]/fb/dbm-steigung
lambdaa = const.value("Planck constant")*const.c/(steigung*const.value("electron volt"))

print('U1 = ', steigung)
print('lambda = ', lambdaa)
#print('k = ', k2)
"""
plt.cla()
plt.clf()
x_plot = np.linspace(0,8)
plt.plot(n, a*1000, 'rx', label='Messwerte')
plt.plot(x_plot, (x_plot*paramsLinear[0]+paramsLinear[1])*1000, 'b-', label='Ausgleichsgerade')
plt.xlim(0,8)
plt.xlabel(r'$n$')
plt.ylabel(r'$U/\si{\volt}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/fig3.pdf')
"""

#c)
print('c)')

Us = 136/1000*fc
Us2 = 115/1000*fc
E_Io1 = Us-k1
E_Io2 = Us2-k1

print('E_Io1 = ', E_Io1)
print('E_Io2 = ', E_Io2)






