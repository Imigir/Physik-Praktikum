﻿from table import makeTable
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
import math

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

#a) Schallgeschwindigkeit Impuls-Echo-Verfahren
l,t,t1 = np.genfromtxt('scripts/data1.txt', unpack=True)
l = l/100
t=t/(2*10**6)

paramsLinear, errorsLinear, sigma_y = linregress(t,l)
steigung = unp.uarray(paramsLinear[0], errorsLinear[0])
achsenAbschnitt = unp.uarray(paramsLinear[1], errorsLinear[1])

#paramsLinear, covar = curve_fit(linear, t, l)
#errorsLinear = np.sqrt(np.diag(covar))
#steigung = unp.uarray(paramsLinear[0], errorsLinear[0])

print('a)')
print('c=', steigung)
print('Abweichung in %:', (paramsLinear[0]-2730)/2730*100)
print('Achsenabschnitt:', achsenAbschnitt)

plt.cla()
plt.clf()
x_plot = np.linspace(0,50)
plt.plot(t*10**6, l*100, 'rx', linewidth=0.8, label='Messwerte')
plt.plot(x_plot, x_plot/10000*paramsLinear[0], 'k-', linewidth=0.8, label='Ausgleichsgerade')
plt.xlabel(r'$t/10^{-6}\si{\second}$')
plt.ylabel(r'$l/10^{-2}\si{\metre}$')
plt.xlim(0,50)
plt.ylim(0,15)
plt.legend(loc="best")
plt.savefig('content/images/Schallgeschwindigkeit.pdf')


makeTable([t1, 0.3+t*2*10**6, t*2*10**6, t*10**6, l*100], r'{'+r'$t_.1/10^{-6}\si{\second}$'+r'} & {'+r'$t_.2/10^{-6}\si{\second}$'+r'} & {'+r'$\Delta t_.{mess}/10^{-6}\si{\second}$'+r'} & {'+r'$\Delta t_.{eff}/10^{-6}\si{\second}$'+r'} & {'+r'$l/10^{-2}\si{\metre}$'+r'}', 'tabSchallgeschwindigkeit', ['S[table-format=1.1]', 'S[table-format=2.1]', 'S[table-format=2.1]', 'S[table-format=2.2]', 'S[table-format=2.2]'], ["%1.1f", "%2.1f", "%2.1f", "%2.2f", "%2.2f"])

#b) Schallgeschwindigkeit Durchschallungsverfahren

l_D,t_D = np.genfromtxt('scripts/data2.txt', unpack=True)
t_D = t_D/1000000
l_D = l_D/100

paramsLinear_D, errorsLinear_D, sigma_y = linregress(t_D,l_D)
steigung_D = unp.uarray(paramsLinear_D[0], errorsLinear_D[0])
achsenAbschnitt_D = unp.uarray(paramsLinear_D[1], errorsLinear_D[1])

#paramsLinear_D, covar_D = curve_fit(linear, t_D, l_D)
#errorsLinear_D = np.sqrt(np.diag(covar_D))
#steigung_D = unp.uarray(paramsLinear_D[0], errorsLinear_D[0])

print('b)')
print('c_D=', steigung_D)
print('Abweichung in %:', (paramsLinear_D[0]-2730)/2730*100)
print('Achsenabschnitt_D:', achsenAbschnitt_D)

plt.cla()
plt.clf()
x_plot = np.linspace(0,50)
plt.plot(t_D*10**6, l_D*100, 'rx', linewidth=0.8, label='Messwerte')
plt.plot(x_plot, x_plot/10000*paramsLinear_D[0], 'k-', linewidth=0.8, label='Ausgleichsgerade')
plt.xlabel(r'$t/10^{-6}\si{\second}$')
plt.ylabel(r'$l/10^{-2}\si{\metre}$')
plt.xlim(0,50)
plt.ylim(0,15)
plt.legend(loc="best")
plt.savefig('content/images/Schallgeschwindigkeit-Durchschallung.pdf')

makeTable([t_D*10**6, l*100], r'{'+r'$\Delta t_.{Durchschallung}/\si{\second}$'+r'} & {'+r'$l/10^{-2}\si{\metre}$'+r'}', 'tabSchallgeschwindigkeitDurchschallung', ['S[table-format=2.1]', 'S[table-format=2.2]'], ["%2.1f", "%2.2f"])


#c) Dämpfung Impuls-Echo-Verfahren

#def efunction(x,a,b):
#   return np.exp(a*x)+b

l_Dae,U =np.genfromtxt('scripts/data3.txt', unpack=True)
l_Dae = l_Dae/100

params_e, errors_e, sigma_y = linregress(l_Dae, np.log(U))
steigung_e = unp.uarray(params_e[0], errors_e[0])
achsenAbschnitt = unp.uarray(params_e[1],errors_e[1])

params_e1, errors_e1, sigma_y = linregress(l_Dae[2:], np.log(U)[2:])
steigung_e1 = unp.uarray(params_e1[0], errors_e1[0])
achsenAbschnitt1 = unp.uarray(params_e1[1],errors_e1[1])

params_e2, errors_e2, sigma_y = linregress(l_Dae[(l_Dae<0.039)|(l_Dae>0.09)], np.log(U[(U>1.36)|(U<1)]))
steigung_e2 = unp.uarray(params_e2[0], errors_e2[0])
achsenAbschnitt2 = unp.uarray(params_e2[1],errors_e2[1])

print('c)')
print('Steigung: ', steigung_e)
print('Achsenabsabschnitt: ', achsenAbschnitt)
print('Dämpfungsfaktor: ',-2*steigung_e)

print('Steigung Interp1: ', steigung_e1)
print('Achsenabsabschnitt Interp1: ', achsenAbschnitt1)
print('Dämpfungsfaktor Interp1: ',-2*steigung_e1)

print('Steigung Interp2: ', steigung_e2)
print('Achsenabsabschnitt Interp2: ', achsenAbschnitt2)
print('Dämpfungsfaktor Interp2: ',-2*steigung_e2)

plt.cla()
plt.clf()
x_plot = np.linspace(0,15)
plt.plot(l_Dae*100, np.log(U), 'rx', linewidth=0.8, label='Messwerte')
plt.plot(x_plot, x_plot/100*params_e[0]+params_e[1], 'k-', linewidth=0.8, label='Ausgleichsgerade')
plt.plot(x_plot, x_plot/100*params_e1[0]+params_e1[1], 'b-', linewidth=0.8, label='Ausgleichsgerade Interpretation1')
plt.plot(x_plot, x_plot/100*params_e2[0]+params_e2[1], 'orange', linewidth=0.8, label='Ausgleichsgerade Interpretation2')
plt.ylabel(r'$\ln(U/\si{\volt})$')
plt.xlabel(r'$l/10^{-2}\si{\metre}$')
plt.xlim(0,15)
#plt.ylim(0,1.5)
plt.legend(loc="best")
plt.savefig('content/images/Daempfung.pdf')

makeTable([l_Dae*100, -(U-1.487)], r'{'+r'$l/10^{-2}\si{\metre}$'+r'} & {'+r'$\Delta U/\si{\volt}$'+r'}', 'tabDaempfung', ['S[table-format=2.2]', 'S[table-format=1.3]'], ["%2.2f", "%1.3f"])


#d) Augenmodelluntersuchung

n, t_A, t_A_eff = np.genfromtxt('scripts/data5.txt', unpack=True)

makeTable([n, t_A_eff], r'{'+r'$n$'+r'} & {'+r'$\Delta t_.A/10^{-6}\si{\second}$'+r'}', 'tabAuge', ['S[table-format=1.0]', 'S[table-format=2.1]'], ["%1.0f", "%2.1f"])

#e) Mehrfachecho und Cepstrum

l_c, dt_c = np.genfromtxt('scripts/data4.txt',unpack=True)

makeTable([l_c, dt_c, dt_c/2, dt_c/20000*2730], r'{'+r'$d_.{mess}/10^{-2}\si{\metre}$'+r'} & {'+r'$\Delta t_.{mess}/10^{-6}\si{\second}$'+r'} & {'+r'$\Delta t_.{eff}/10^{-6}\si{\second}$'+r'} & {'+r'$d_.{exp}/10^{-2}\si{\metre}$'+r'}', 'tabMehrfachecho', ['S[table-format=1.2]', 'S[table-format=2.1]', 'S[table-format=2.2]', 'S[table-format=1.2]'], ["%1.2f", "%2.1f", "%2.2f", "%1.2f"])
