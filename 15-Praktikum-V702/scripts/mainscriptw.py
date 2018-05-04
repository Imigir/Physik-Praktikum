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
import math as ma

"""BackwardsVNominal = []
BackwardsVStd = []
for value in BackwardsV:
     BackwardsVNominal.append(unp.nominal_values(value))
     BackwardsVStd.append(unp.std_devs(value))
BackwardsVNominal = np.array(BackwardsVNominal)
BackwardsVStd = np.array(BackwardsVStd)

einfacher:
BackwardsVNominal = unp.nominal_values(BackwardsV)
BackwardsVStd = unp.std_devs(BackwardsV)

makeTable([Gaenge, ForwardsVNominal, ForwardsVStd, ], r'{Gang} & \multicolumn{2}{c}{$v_\text{v}/\si[per-mode=reciprocal]{\centi\meter\per\second}$} & ', 'name', ['S[table-format=2.0]', 'S[table-format=2.3]', ' @{${}\pm{}$} S[table-format=1.3]', ], ["%2.0f", "%2.3f", "%2.3f",])

[per-mode=reciprocal],[table-format=2.3,table-figures-uncertainty=1]

unp.uarray(np.mean(), stats.sem())
unp.uarray(*avg_and_sem(values)))
unp.uarray(*weighted_avg_and_sem(unp.nominal_values(bneuDiff), 1/unp.std_devs(bneuDiff)))

plt.cla()
plt.clf()
plt.plot(ForwardsVNominal*100, DeltaVForwardsNominal, 'gx', label='Daten mit Bewegungsrichtung aufs Mikrofon zu')
plt.plot(BackwardsVNominal*100, DeltaVBackwardsNominal, 'rx', label='Daten mit Bewegungsrichtung vom Mikrofon weg')
plt.ylim(0, line(t[-1], *params)+0.1)
plt.xlim(0, t[-1]*100)
plt.xlabel(r'$v/\si{\centi\meter\per\second}$')
plt.ylabel(r'$\Delta f / \si{\hertz}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/'+'VgegenDeltaV')
"""

#a)
#1 Vanadium
t,N = np.genfromtxt('scripts/data1.txt', unpack = True)
N1 = N[N>8]
N_0 = 223/900
N_err = np.sqrt(N-N_0*30)/30
N_err2 = np.sqrt(N1-N_0*30)/30
N = N/30-N_0
N1 = N1/30-N_0
N_log = np.log(N1)
N_log_err = [np.log(N1+N_err2)-np.log(N1), np.log(N1)-np.log(N1-N_err2)]
N = unp.uarray(N, N_err)
N1 = unp.uarray(N1, N_err2)
t = t*30
t1 = t[t!=22*30]

#print('N=', N)

paramsLinear, errorsLinear, sigma_y = linregress(t1, N_log)

steigung = unp.uarray(paramsLinear[0], errorsLinear[0])
achsenAbschnitt = unp.uarray(paramsLinear[1], errorsLinear[1])

print('Lambda=', -steigung)
print('N0=', unp.exp(achsenAbschnitt))
print('tau1=', -np.log(2)/steigung)


plt.cla()
plt.clf()
x_plot = np.linspace(0,30*30+20)
plt.errorbar(t1, N_log, yerr=[N_log_err[0],N_log_err[1]], fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot, x_plot*paramsLinear[0]+paramsLinear[1], 'k-', linewidth=0.8, label='Ausgleichsgerade')
plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$\ln(N/\si{\becquerel})$')
plt.xlim(0,30*30+20)
plt.legend(loc="best")
plt.savefig('content/images/VanadiumLog.pdf')

plt.cla()
plt.clf()
x_plot = np.linspace(0,30*30+20)
plt.errorbar(t1, noms(N1), yerr=stds(N1), fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.errorbar(t[21], noms(N)[21], yerr=stds(N)[21], color='grey', fmt='x', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='lightgrey',barsabove=True ,label='ungenutzter Messwert')
plt.plot(x_plot, np.exp(x_plot*paramsLinear[0]+paramsLinear[1]), 'k-', linewidth=0.8, label='Ausgleichskurve')
plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$N/\si{\becquerel}$')
plt.xlim(0,30*30+20)
plt.legend(loc="best")
plt.savefig('content/images/Vanadium.pdf')


makeTable([t1, noms(N1), stds(N1), N_log, N_log_err[0], N_log_err[1]], r'{'+r'$t/\si{\second}$'+r'} & {'+r'$N_.V/\si{\becquerel}$'+r'} & {'+r'$\sigma_{N_.V}/\si{\becquerel}$'+r'} & {'+r'$\ln\left(N_.V/\si{\becquerel}\right)$'+r'} & {'+r'$\left|\ln\left(\frac{N_.V+\sigma_{N_.V}}{N}\right)\right|$'+r'} & {'+r'$\left|\ln\left(\frac{N_.V}{N_.V-\sigma_{N_.V}}\right)\right|$'+r'}', 'tabVanadium', ['S[table-format=3.0]', 'S[table-format=1.1]', 'S[table-format=1.1]', 'S[table-format=1.2]', 'S[table-format=1.2]', 'S[table-format=1.2]'], ["%3.0f", "%1.1f", "%1.1f", "%1.2f", "%1.2f", "%1.2f"])

#2 Rhodium
t2,N2 = np.genfromtxt('scripts/data2.txt', unpack = True)
N2_err = np.sqrt(N2-N_0*10)/10
N2 = N2/10-N_0
N2_log = np.log(N2[:44])
N2_err2 = N2-N2_err
N2_log_err = [np.log(N2[:44]+N2_err[:44])-np.log(N2[:44]), np.log(N2[:44])-np.log(N2_err2[:44])]
N2 = unp.uarray(N2, N2_err)
t2 = t2*10

paramsLinear2, errorsLinear2, sigma_y = linregress(t2[25:44], N2_log[25:44])
steigung2 = unp.uarray(paramsLinear2[0], errorsLinear2[0])
achsenAbschnitt2 = unp.uarray(paramsLinear2[1], errorsLinear2[1])

paramsLinear1, errorsLinear1, sigma_y = linregress(t2[0:20], np.log(noms(N2)[0:20]-np.exp(t2[0:20]*paramsLinear2[0]+paramsLinear2[1])))
steigung1 = unp.uarray(paramsLinear1[0], errorsLinear1[0])
achsenAbschnitt1 = unp.uarray(paramsLinear1[1], errorsLinear1[1])

x_plot = np.linspace(0,60*10+10)
N_t = np.exp(x_plot*paramsLinear1[0]+paramsLinear1[1])+np.exp(x_plot*paramsLinear2[0]+paramsLinear2[1])
N21 = N2[0:20]-np.exp(t2[0:20]*paramsLinear2[0]+paramsLinear2[1])
N21_log_err = [np.log(noms(N21)+stds(N21))-np.log(noms(N21)), np.log(noms(N21))-np.log(noms(N21)-stds(N21))]

print('Lambda21=', -steigung1)
print('N021=', unp.exp(achsenAbschnitt1))
print('tau21=', -np.log(2)/steigung1)
print('Lambda22=', -steigung2)
print('N022=', unp.exp(achsenAbschnitt2))
print('tau22=', -np.log(2)/steigung2)

plt.cla()
plt.clf()
plt.errorbar(t2[25:44], N2_log[25:44], yerr=[N2_log_err[0][25:44],N2_log_err[1][25:44]], fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot, x_plot*paramsLinear2[0]+paramsLinear2[1], 'orange', linewidth=0.8, label='Ausgleichsgerade2')
plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$\ln(N_\mathrm{Rh_\mathrm{104i}}/\si{\becquerel})$')
plt.xlim(25*10,44*10+10)
plt.legend(loc="best")
plt.savefig('content/images/RhodiumLog2.pdf')

plt.cla()
plt.clf()
plt.errorbar(t2[0:20], np.log(noms(N2)[0:20]-np.exp(t2[0:20]*paramsLinear2[0]+paramsLinear2[1])), yerr=[N21_log_err[0][0:20],N21_log_err[1][0:20]], fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot, x_plot*paramsLinear1[0]+paramsLinear1[1], linewidth=0.8, label='Ausgleichsgerade1')
plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$\ln((N-N_\mathrm{Rh_\mathrm{104i}})/\si{\becquerel})$')
plt.xlim(0,20*10+10)
plt.legend(loc="best")
plt.savefig('content/images/RhodiumLog1.pdf')

plt.cla()
plt.clf()
plt.errorbar(t2[0:44], N2_log[0:44], yerr=[N2_log_err[0][0:44],N2_log_err[1][0:44]], fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot, x_plot*paramsLinear1[0]+paramsLinear1[1], linewidth=0.8, label='Ausgleichsgerade1')
plt.plot(x_plot, x_plot*paramsLinear2[0]+paramsLinear2[1], linewidth=0.8, label='Ausgleichsgerade2')
plt.plot(x_plot, np.log(N_t), 'k-', linewidth=0.8, label='N(t)')
plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$\ln(N/\si{\becquerel})$')
plt.xlim(0,44*10+10)
plt.legend(loc="best")
plt.savefig('content/images/RhodiumLog3.pdf')

plt.cla()
plt.clf()
x_plot = np.linspace(0,60*10+10)
plt.errorbar(t2[:44], noms(N2)[:44], yerr=stds(N2)[:44], fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.errorbar(t2[44:], noms(N2)[44:], yerr=stds(N2)[44:], color='grey', fmt='x', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='lightgrey',barsabove=True ,label='ungenutzte Messwerte')
plt.plot(x_plot, N_t, 'k-', linewidth=0.8, label='N(t)')
plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$N/\si{\becquerel}$')
plt.xlim(0,60*10+10)
plt.legend(loc="best")
plt.savefig('content/images/Rhodium.pdf')

makeTable([t2[0:29], noms(N2)[0:29], stds(N2)[0:29], N2_log[0:29], N2_log_err[0][0:29], N2_log_err[1][0:29]], r'{'+r'$t/\si{\second}$'+r'} & {'+r'$N_.{Rh}/\si{\becquerel}$'+r'} & {'+r'$\sigma_{N_.{Rh}}/\si{\becquerel}$'+r'} & {'+r'$\ln\left(N_.{Rh}/\si{\becquerel}\right)$'+r'} & {'+r'$\left|\ln\left(\frac{N_.{Rh}+\sigma_{N_.{Rh}}}{N_.{Rh}}\right)\right|$'+r'} & {'+r'$\left|\ln\left(\frac{N_.{Rh}}{N_.{Rh}-\sigma_{N_.{Rh}}}\right)\right|$'+r'}', 'tabRhodium1', ['S[table-format=3.0]', 'S[table-format=1.1]', 'S[table-format=1.1]', 'S[table-format=1.2]', 'S[table-format=1.2]', 'S[table-format=1.2]'], ["%3.0f", "%1.1f", "%1.1f", "%1.2f", "%1.2f", "%1.2f"])
makeTable([t2[29:44], noms(N2)[29:44], stds(N2)[29:44], N2_log[29:44], N2_log_err[0][29:44], N2_log_err[1][29:44]], r'{'+r'$t/\si{\second}$'+r'} & {'+r'$N_.{Rh}/\si{\becquerel}$'+r'} & {'+r'$\sigma_{N_.{Rh}}/\si{\becquerel}$'+r'} & {'+r'$\ln\left(N_.{Rh}/\si{\becquerel}\right)$'+r'} & {'+r'$\left|\ln\left(\frac{N_.{Rh}+\sigma_{N_.{Rh}}}{N_.{Rh}}\right)\right|$'+r'} & {'+r'$\left|\ln\left(\frac{N_.{Rh}}{N_.{Rh}-\sigma_{N_.{Rh}}}\right)\right|$'+r'}', 'tabRhodium2', ['S[table-format=3.0]', 'S[table-format=1.1]', 'S[table-format=1.1]', 'S[table-format=1.2]', 'S[table-format=1.2]', 'S[table-format=1.2]'], ["%3.0f", "%1.1f", "%1.1f", "%1.2f", "%1.2f", "%1.2f"])
