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
# unp.uarray(*avg_and_sem(values))
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


def linear(x, A, B):
    return A*x+B
	
#Konstanten
mu_0 = const.mu_0
g = const.g
pi = const.pi

#SpulenWerte
N = 195
x = 0.138/2
R = 0.109

#KugelWerte
m_K = 142/1000
r_K = 5.33/200
J_K = 2/5*m_K*r_K**2

#Gravitation
r,I = np.genfromtxt("scripts/data1.txt", unpack=True)
r = (r+1.4+5.33/2)*0.01
m = 1.4/1000

B = (N*mu_0*I*R**2)/((R**2+x**2)**(3/2))
	
paramsLinear, errorsLinear, sigma_y = linregress(B, r)

steigung = unp.uarray(paramsLinear[0], errorsLinear[0])
achsenAbschnitt = unp.uarray(paramsLinear[1], errorsLinear[1])

m_Dipol = m*g*steigung

print('Gravitation:')
print('Steigung =', steigung)
print('Achsenabschnitt =', achsenAbschnitt)
print('m_Dipol =', m_Dipol)

plt.cla()
plt.clf()
x_plot = np.linspace(0,5)
plt.plot(B*1000, r*100, 'rx', label ="Messwerte")
plt.plot(x_plot, linear(x_plot/1000, *paramsLinear)*100, 'b-', label='Ausgleichsgerade')
plt.xlim(1.5,3.25)
plt.xlabel(r'$B/\si{\milli\tesla}$')
plt.ylabel(r'$r / \si{\centi\metre}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
plt.savefig("content/images/Gravitation")


makeTable([r*100, I, B*1000], r'{'+r'$r/\si{\centi\metre}$'+r'} & {'+r'$I/\si[per-mode=reciprocal]{\ampere}$'+r'} & {'+r'$B/\si[per-mode=reciprocal]{\milli\tesla}$'+r'}', 'taba', ['S[table-format=1.1]', 'S[table-format=1.2]', 'S[table-format=1.2]'], ["%1.1f", "%1.2f", "%1.2f"])

#Schwingungsdauer
I,T = np.genfromtxt("scripts/data2.txt", unpack=True)
T = T/10

B = (N*mu_0*I*R**2)/((R**2+x**2)**(3/2))

paramsLinear, errorsLinear, sigma_y = linregress(1/B, T**2)

steigung = unp.uarray(paramsLinear[0], errorsLinear[0])
achsenAbschnitt = unp.uarray(paramsLinear[1], errorsLinear[1])

m_Dipol = (4*pi**2*J_K)/steigung

print('Schwingungsdauer:')
print('J_K =', J_K)
print('Steigung =', steigung)
print('Achsenabschnitt =', achsenAbschnitt)
print('m_Dipol =', m_Dipol)

plt.cla()
plt.clf()
x_plot = np.linspace(0,1500)
plt.plot(1/B, T**2, 'rx', label ="Messwerte")
plt.plot(x_plot, linear(x_plot, *paramsLinear), 'b-', label='Ausgleichsgerade')
plt.xlim(150,1500)
plt.xlabel(r'$B^{-1}/\si{\per\tesla}$')
plt.ylabel(r'$T^2 / \si{\second\squared}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
plt.savefig("content/images/Schwingungsdauer")

makeTable([I, B*1000, T], r'{'+r'$I/\si{\ampere}$'+r'} & {'+r'$B/\si[per-mode=reciprocal]{\milli\tesla}$'+r'} & {'+r'$T/\si[per-mode=reciprocal]{\second}$'+r'}', 'tabb', ['S[table-format=1.1]', 'S[table-format=1.1]', 'S[table-format=2.2]'], ["%1.1f", "%1.1f", "%2.2f"])

#Präzession
I,T = np.genfromtxt("scripts/data3.txt", unpack=True)

In = []
Tn = []
Ts = []
for i in range(10):
	Ttn, Tts = avg_and_sem(T[3*i:3*i+3])
	In.append(I[i*3])
	Tn.append(Ttn)
	Ts.append(Tts)
I = np.array(In)
T = unp.uarray(Tn,Ts)
Tn = np.array(Tn)

B = (N*mu_0*I*R**2)/((R**2+x**2)**(3/2))
L_K = J_K*2*pi*5

paramsLinear, errorsLinear, sigma_y = linregress(B, 1/Tn)


steigung = unp.uarray(paramsLinear[0], errorsLinear[0])
achsenAbschnitt = unp.uarray(paramsLinear[1], errorsLinear[1])

m_Dipol = 2*pi*L_K*steigung

print('Präzission:')
print('L_K =', L_K)
print('Steigung =', steigung)
print('Achsenabschnitt =', achsenAbschnitt)
print('m_Dipol =', m_Dipol)

plt.cla()
plt.clf()
x_plot = np.linspace(0,6)
plt.plot(B*1000, 1/Tn, 'rx', label ="Messwerte")
plt.plot(x_plot, linear(x_plot/1000, *paramsLinear), 'b-', label='Ausgleichsgerade')
plt.xlim(0.5,5.75)
plt.xlabel(r'$B/\si{\milli\tesla}$')
plt.ylabel(r'$T^{-1} / \si{\per\second}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
plt.savefig("content/images/Praezession")

makeTable([I, B*1000, Tn, Ts], r'{'+r'$I/\si{\ampere}$'+r'} & {'+r'$B/\si[per-mode=reciprocal]{\milli\tesla}$'+r'} & \multicolumn{2}{c}{'+r'$T/\si[per-mode=reciprocal]{\second}$'+r'}', 'tabc', ['S[table-format=1.1]', 'S[table-format=1.1]', 'S[table-format=2.2]', ' @{${}\pm{}$} S[table-format=1.2]'], ["%1.1f", "%1.1f", "%2.2f", "%1.2f"])

