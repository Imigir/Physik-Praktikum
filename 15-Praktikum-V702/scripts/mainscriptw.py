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
"""
BackwardsVNominal = []
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
N_0 = 223/900
N_err = np.sqrt(N-N_0*30)/30
N = N/30-N_0
N_log = np.log(N)
N_log_err = [np.log(N+N_err)-np.log(N), np.log(N)-np.log(N-N_err)]
N = unp.uarray(N, N_err)
t = t*30

#print('N=', N)
"""
paramsLinear, errorsLinear, sigma_y = linregress(t, N_log)

steigung = unp.uarray(paramsLinear[0], errorsLinear[0])
achsenAbschnitt = unp.uarray(paramsLinear[1], errorsLinear[1])

print('Lambda=', -steigung)
print('N0=', unp.exp(achsenAbschnitt))
print('tau1=', -np.log(2)/steigung)


plt.cla()
plt.clf()
x_plot = np.linspace(0,30*30+20)
plt.errorbar(t, N_log, yerr=[N_log_err[0],N_log_err[1]], fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot, x_plot*paramsLinear[0]+paramsLinear[1], 'k-', linewidth=0.8, label='Ausgleichsgerade')
plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$\ln(N/\si{\becquerel})$')
plt.xlim(0,30*30+20)
plt.legend(loc="best")
plt.savefig('content/images/VanadiumLog.pdf')

plt.cla()
plt.clf()
x_plot = np.linspace(0,30*30+20)
plt.errorbar(t, noms(N), yerr=stds(N), fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot, np.exp(x_plot*paramsLinear[0]+paramsLinear[1]), 'k-', linewidth=0.8, label='Ausgleichskurve')
plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$N/\si{\becquerel}$')
plt.xlim(0,30*30+20)
plt.legend(loc="best")
plt.savefig('content/images/Vanadium.pdf')


makeTable([t, noms(N), stds(N), N_log, N_log_err[0], N_log_err[1]], r'{'+r'$t/\si{\second}$'+r'} & {'+r'$N_.V/\si{\becquerel}$'+r'} & {'+r'$\sigma_{N_.V}/\si{\becquerel}$'+r'} & {'+r'$\ln\left(N_.V/\si{\becquerel}\right)$'+r'} & {'+r'$\ln\left(\frac{N_.V+\sigma_{N_.V}}{N}\right)$'+r'} & {'+r'$\ln\left(\frac{N_.V}{N_.V-\sigma_{N_.V}}\right)$'+r'}', 'tabVanadium', ['S[table-format=3.0]', 'S[table-format=1.1]', 'S[table-format=1.1]', 'S[table-format=1.2]', 'S[table-format=1.2]', 'S[table-format=1.2]'], ["%3.0f", "%1.1f", "%1.1f", "%1.2f", "%1.2f", "%1.2f"])
"""
#2 Rhodium
t2,N2 = np.genfromtxt('scripts/data2.txt', unpack = True)
N2_err = np.sqrt(N2-N_0*10)/10
N2 = N2/10-N_0
N2_log = np.log(N2)
N2_log_err = [np.log(N2+N2_err)-np.log(N2), np.log(N2)-np.log(N2-N2_err)]
N2 = unp.uarray(N2, N2_err)
t2 = t2*10

paramsLinear2, errorsLinear2, sigma_y = linregress(t2[25:45], N2_log[25:45])
steigung2 = unp.uarray(paramsLinear2[0], errorsLinear2[0])
achsenAbschnitt2 = unp.uarray(paramsLinear2[1], errorsLinear2[1])

paramsLinear1, errorsLinear1, sigma_y = linregress(t2[0:20], np.log(noms(N2)[0:20]-np.exp(t2[0:20]*paramsLinear2[0]+paramsLinear2[1])))
steigung1 = unp.uarray(paramsLinear1[0], errorsLinear1[0])
achsenAbschnitt1 = unp.uarray(paramsLinear1[1], errorsLinear1[1])

x_plot = np.linspace(0,60*10+10)
N_t = np.exp(x_plot*paramsLinear1[0]+paramsLinear1[1])+np.exp(x_plot*paramsLinear2[0]+paramsLinear2[1])

print('Lambda21=', -steigung1)
print('N021=', unp.exp(achsenAbschnitt1))
print('tau21=', -np.log(2)/steigung1)
print('Lambda22=', -steigung2)
print('N022=', unp.exp(achsenAbschnitt2))
print('tau22=', -np.log(2)/steigung2)

plt.cla()
plt.clf()
plt.errorbar(t2, N2_log, yerr=[N2_log_err[0],N2_log_err[1]], fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot, x_plot*paramsLinear1[0]+paramsLinear1[1], linewidth=0.8, label='Ausgleichsgerade1')
plt.plot(x_plot, x_plot*paramsLinear2[0]+paramsLinear2[1], linewidth=0.8, label='Ausgleichsgerade2')
plt.plot(x_plot, np.log(N_t), 'k-', linewidth=0.8, label='N(t)')
plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$\ln(N/\si{\becquerel})$')
plt.xlim(0,44*10+10)
plt.legend(loc="best")
plt.savefig('content/images/RhodiumLog.pdf')

plt.cla()
plt.clf()
x_plot = np.linspace(0,60*10+10)
plt.errorbar(t2, noms(N2), yerr=stds(N2), fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot, N_t, 'k-', linewidth=0.8, label='N(t)')
plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$N/\si{\becquerel}$')
plt.xlim(0,44*10+10)
plt.legend(loc="best")
plt.savefig('content/images/Rhodium.pdf')

makeTable([t2[0:29], noms(N2)[0:29], stds(N2)[0:29], N2_log[0:29], N2_log_err[0][0:29], N2_log_err[1][0:29]], r'{'+r'$t/\si{\second}$'+r'} & {'+r'$N_.{Rh}/\si{\becquerel}$'+r'} & {'+r'$\sigma_{N_.{Rh}}/\si{\becquerel}$'+r'} & {'+r'$\ln\left(N_.{Rh}/\si{\becquerel}\right)$'+r'} & {'+r'$\ln\left(\frac{N_.{Rh}+\sigma_{N_.{Rh}}}{N_.{Rh}}\right)$'+r'} & {'+r'$\ln\left(\frac{N_.{Rh}}{N_.{Rh}-\sigma_{N_.{Rh}}}\right)$'+r'}', 'tabRhodium1', ['S[table-format=3.0]', 'S[table-format=1.1]', 'S[table-format=1.1]', 'S[table-format=1.2]', 'S[table-format=1.2]', 'S[table-format=1.2]'], ["%3.0f", "%1.1f", "%1.1f", "%1.2f", "%1.2f", "%1.2f"])
makeTable([t2[29:45], noms(N2)[29:45], stds(N2)[29:45], N2_log[29:45], N2_log_err[0][29:45], N2_log_err[1][29:45]], r'{'+r'$t/\si{\second}$'+r'} & {'+r'$N_.{Rh}/\si{\becquerel}$'+r'} & {'+r'$\sigma_{N_.{Rh}}/\si{\becquerel}$'+r'} & {'+r'$\ln\left(N_.{Rh}/\si{\becquerel}\right)$'+r'} & {'+r'$\ln\left(\frac{N_.{Rh}+\sigma_{N_.{Rh}}}{N_.{Rh}}\right)$'+r'} & {'+r'$\ln\left(\frac{N_.{Rh}}{N_.{Rh}-\sigma_{N_.{Rh}}}\right)$'+r'}', 'tabRhodium2', ['S[table-format=3.0]', 'S[table-format=1.1]', 'S[table-format=1.1]', 'S[table-format=1.2]', 'S[table-format=1.2]', 'S[table-format=1.2]'], ["%3.0f", "%1.1f", "%1.1f", "%1.2f", "%1.2f", "%1.2f"])
