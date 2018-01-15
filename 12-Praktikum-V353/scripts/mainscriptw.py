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


pi = const.pi
#1
print('1:')
t, U = np.genfromtxt('scripts/data1.txt', unpack=True)
t = (t-4.36)/1000
U_0 = 49.6

paramsLinear, errorsLinear, sigma_y = linregress(t, np.log(U/U_0))

steigung = unp.uarray(paramsLinear[0], errorsLinear[0])
achsenAbschnitt = unp.uarray(paramsLinear[1], errorsLinear[1])

print('Steigung =', steigung)
print('Achsenabschnitt =', achsenAbschnitt)
print('RC =', 1/steigung)

plt.cla()
plt.clf()
x_plot = np.linspace(-1,5)
plt.plot(t*1000, np.log(U/U_0), 'rx', label ="Messwerte")
plt.plot(x_plot, x_plot/1000*paramsLinear[0], 'b-', label='Ausgleichsgerade')
plt.xlim(-0.3,3.8)
#plt.ylim(0,46)
plt.xlabel(r'$t/10^{-3}\si{\second}$')
plt.ylabel(r'$\mathrm{log}\left(\frac{U_\mathrm{C}}{U_0}\right)$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
plt.savefig("content/images/Graph1")

makeTable([t*1000, U, np.log(U/U_0)], r'{'+r'$t/10^{-3}\si{\second}$'+r'} & {'+r'$U_\mathrm{C}/\si{\volt}$'+r'} & {'+r'$\mathrm{log}\left(\frac{U_\mathrm{C}}{U_0}\right)$'+r'}', 'taba', ['S[table-format=1.2]', 'S[table-format=2.1]', 'S[table-format=2.1]'], ["%1.2f", "%2.1f", "%2.1f"])

#2
print('2:')
def Amplitude(x, c):
	return 1/np.sqrt(1+x**2*c**2)

f, U, a = np.genfromtxt('scripts/data2.txt', unpack=True)
a = a/1000
U_0 = 96
 
params, covar = curve_fit(Amplitude, f, U/U_0)
RC = unp.uarray(params[0], np.sqrt(covar[0][0]))
print('RC =', RC)

plt.cla()
plt.clf()
x_plot = np.logspace(0,5,100)
plt.plot(f, U/U_0, 'rx', label ="Messwerte")
plt.plot(x_plot, Amplitude(x_plot, *params), 'b-', label='Ausgleichskurve')
plt.xscale('log')
plt.xlim(6,15000)
#plt.ylim(0,46)
plt.xlabel(r'$f/\si{\hertz}$')
plt.ylabel(r'$\frac{A}{U_0}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
plt.savefig("content/images/Graph2")

#3
print('3:')
def Phase(x, c):
	return np.arctan(-x*c)

b = 1/f
phi = a/b*2*pi

params, covar = curve_fit(Phase, f, phi)
RC = unp.uarray(params[0], np.sqrt(covar[0][0]))
print('RC =', RC)

plt.cla()
plt.clf()
x_plot = np.logspace(0,5,100)
plt.plot(f, phi, 'rx', label ="Messwerte")
plt.plot(x_plot, Phase(x_plot, *params), 'b-', label='Ausgleichskurve')
plt.xscale('log')
plt.xlim(6,15000)
#plt.ylim(0,46)
plt.yticks( [0, pi/4, pi/2],[r'$0$', r'$\pi/4$', r'$\pi/2$'])
plt.xlabel(r'$f/\si{\hertz}$')
plt.ylabel(r'$\phi/\si{\radian}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
plt.savefig("content/images/Graph3")

makeTable([f, U, U/U_0, a*10000, phi], r'{'+r'$f/\si{\hertz}$'+r'} & {'+r'$A/\si{\volt}$'+r'} & {'+r'$\frac{A}{U_0}$'+r'} & {'+r'$a/10^{-4}\si{\second}$'+r'} & {'+r'$\phi/\si{\radian}$'+r'}', 'tabb', ['S[table-format=5.0]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=1.2]', 'S[table-format=1.2]'], ["%5.0f", "%2.2f", "%2.2f", "%1.2f", "%1.2f"])

#4
def Amplitude(x):
	return np.cos(Phase(x,params[0]))

plt.cla()
plt.clf()
plt.subplot(111, projection='polar')
x_plot = np.logspace(0,5,100)
plt.plot(phi,U/U_0, 'rx', label ="Messwerte")
plt.plot(Phase(x_plot,*params), Amplitude(x_plot), 'b-', label='Theoriekurve')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(framealpha=1, frameon=True)
plt.savefig("content/images/Graph4")
