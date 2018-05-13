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


#a)
#A-Scan


n, l_oben, dt_oben, l_unten, dt_unten= np.genfromtxt('scripts/data1.txt',unpack=True)
n_4, dt_oben_4, dt_unten_4 = np.genfromtxt('scripts/data1.2.txt', unpack=True)
l_oben_exp = dt_oben*2.73/2
l_oben_exp_4 = dt_oben_4*2.73/2
l_unten_exp = dt_unten*2.73/2
l_unten_exp_4 = dt_unten_4*2.73/2
durchmesser = 80.3-l_oben-l_unten
durchmesser_exp = 2.73*(58.69-dt_oben-dt_unten)/2
durchmesser_exp_4 = 2.73*(58.69-dt_oben_4-dt_unten_4)/2
loeschen =[2,3,4,5,6,7,8,9,10]
durchmesser_4 = np.delete(durchmesser, loeschen)
error_2 = (durchmesser_exp/durchmesser-1)*100
error_4 = (durchmesser_exp_4/durchmesser_4-1)*100
print(error_2)
print(error_4)

#makeTable([n, l_oben, dt_oben, l_oben_exp, l_unten, dt_unten, l_unten_exp, durchmesser, durchmesser_exp, error_2], r'{'+r'$n$'+r'} & {'+r'$l_.{oben}/10^{-3}\si{\metre}$'+r'} & {'+r'$\Delta t_.{oben}/10^{-6}\si{\second}$'+r'} & {'+r'$l_.{oben_.{exp}}/10^{-3}\si{\metre}$'+r'} & {'+r'$l_.{unten}/10^{-3}\si{\metre}$'+r'} & {'+r'$\Delta t_.{unten}/10^{-6}\si{\second}$'+r'} & {'+r'$l_.{unten_.{exp}}/10^{-3}\si{\metre}$'+r'} & {'+r'$d/10^{-3}\si{\metre}$'+r'} & {'+r'$d_.{exp}/10^{-3}\si{\metre}$'+r'} & {'+r'$\Delta d/\%$'+r'}', 'tabAScan2', ['S[table-format=1.0]', 'S[table-format=2.1]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.1]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.1]'], ["%1.0f", "%2.1f", "%2.2f", "%2.2f", "%2.1f", "%2.2f", "%2.2f", "%2.2f", "%2.2f","%2.1f"])
#makeTable([n_4, dt_oben_4, l_oben_exp_4, dt_unten_4, l_unten_exp_4, durchmesser_exp_4, error_4], r'{'+r'$n$'+r'} & {'+r'$\Delta t_.{oben_.{4\si{\mega\hertz}}}/10^{-6}\si{\second}$'+r'} & {'+r'$l_.{oben_.{4\si{\mega\hertz} } }/10^{-3}\si{\metre}$'+r'} & {'+r'$\Delta t_.{unten_.{4\si{\mega\hertz} } }/10^{-6}\si{\second}$'+r'} & {'+r'$l_.{unten_.{4\si{\mega\hertz} } }/10^{-3}\si{\metre}$'+r'} & {'+r'$d_.{exp_.{4\si{\mega\hertz} } }/10^{-3}\si{\metre}$'+r'} & {'+r'$\Delta d_.4/\%$'+r'}', 'tabAScan4',['S[table-format=1.0]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.1]'], ["%1.0f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%2.1f"])

#Referenzwerte
makeTable([n, l_oben, l_unten, durchmesser], r'{'+r'$n$'+r'} & {'+r'$l_.{oben}/10^{-3}\si{\metre}$'+r'} & {'+r'$l_.{unten}/10^{-3}\si{\metre}$'+r'} & {'+r'$d/10^{-3}\si{\metre}$'+r'}', 'tabAScan2MHzRef', ['S[table-format=1.0]', 'S[table-format=2.1]', 'S[table-format=2.1]', 'S[table-format=2.2]'], ["%1.0f", "%2.1f", "%2.1f", "%2.2f"])
#2MHz
makeTable([n, dt_oben, l_oben_exp, dt_unten, l_unten_exp, durchmesser_exp], r'{'+r'$n$'+r'} & {'+r'$\Delta t_.{oben_.{A}}/10^{-6}\si{\second}$'+r'} & {'+r'$l_.{oben_.{A}}/10^{-3}\si{\metre}$'+r'} & {'+r'$\Delta t_.{unten_.{A}}/10^{-6}\si{\second}$'+r'} & {'+r'$l_.{unten_.{A}}/10^{-3}\si{\metre}$'+r'} & {'+r'$d_.{A}/10^{-3}\si{\metre}$'+r'}', 'tabAScan2MHz', ['S[table-format=1.0]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]'], ["%1.0f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%2.2f"])
#4MHz
makeTable([n_4, dt_oben_4, l_oben_exp_4, dt_unten_4, l_unten_exp_4, durchmesser_exp_4], r'{'+r'$n$'+r'} & {'+r'$\Delta t_.{o_.{4\si{\mega\hertz}}}/10^{-6}\si{\second}$'+r'} & {'+r'$l_.{o_.{4\si{\mega\hertz}}}/10^{-3}\si{\metre}$'+r'} & {'+r'$\Delta t_.{u_.{4\si{\mega\hertz}}}/10^{-6}\si{\second}$'+r'} & {'+r'$l_.{u_.{4\si{\mega\hertz}}}/10^{-3}\si{\metre}$'+r'} & {'+r'$d_.{4\si{\mega\hertz}}/10^{-3}\si{\metre}$'+r'}', 'tabAScan4MHz', ['S[table-format=1.0]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]'], ["%1.0f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%2.2f"])


#b)
#B-Scan

n, dt_oben_b, dt_unten_b = np.genfromtxt('scripts/data2.txt', unpack=True)
l_oben_b = dt_oben_b*2.73/2
l_unten_b = dt_unten_b*2.73/2
durchmesser_exp_b = 2.73*(58.69-dt_oben_b-dt_unten_b)/2
error_b=(durchmesser_exp_b/durchmesser-1)*100
print('Error B:', error_b)
makeTable([n, dt_oben_b, l_oben_b, dt_unten_b, l_unten_b, durchmesser_exp_b], r'{'+r'$n$'+r'} & {'+r'$\Delta t_.{oben_.{B} }/10^{-6}\si{\second}$'+r'} & {'+r'$l_.{oben_.{B}}/10^{-3}\si{\metre}$'+r'} & {'+r'$\Delta t_.{unten_.{B}}/10^{-6}\si{\second}$'+r'} & {'+r'$l_.{unten_.{B}}/10^{-3}\si{\metre}$'+r'} & {'+r'$d_.{B}/10^{-3}\si{\metre}$'+r'}', 'tabBScan', ['S[table-format=1.0]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]'], ["%1.0f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%2.2f"])

#Fehler
makeTable([n, durchmesser, durchmesser_exp, durchmesser_exp_b, error_2, error_b], r'{'+r'$n$'+r'} & {'+r'$d/10^{-3}\si{\metre}$'+r'} & {'+r'$d_.{A}/10^{-3}\si{\metre}$'+r'} & {'+r'$d_.{B}/10^{-3}\si{\metre}$'+r'} & {'+r'$\sigma_{d_.{A}}/\%$'+r'} & {'+r'$\sigma_{d_.{B}}/\%$'+r'}', 'tabAScan2MHzFehler', ['S[table-format=1.0]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=1.2]', 'S[table-format=2.1]', 'S[table-format=3.1]'], ["%1.0f", "%2.2f", "%2.2f", "%1.2f","%2.1f","%3.1f"])
makeTable([n_4, durchmesser, durchmesser_exp_4, error_4], r'{'+r'$n$'+r'} & {'+r'$d/10^{-3}\si{\metre}$'+r'} & {'+r'$d_.{4\si{\mega\hertz}}/10^{-3}\si{\metre}$'+r'} & {'+r'$\sigma_{d_.{4\si{\mega\hertz}}}/\%$'+r'}', 'tabAScan4MHzFehler', ['S[table-format=1.0]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.1]'], ["%1.0f", "%2.2f", "%2.2f","%2.1f"])


#c)
#Herz
n, dt, hs = np.genfromtxt('scripts/data3.txt', unpack=True)
h = hs*1.485/2

makeTable([n, dt, hs , h], r'{'+r'$n$'+r'} & {'+r'$\Delta t_.{n\rightarrow n+1}/\si{\second}$'+r'} &{'+r'$\Delta t_.{h}/10^{-6}\si{\second}$'+r'} & {'+r'$h/10^{-3}\si{\metre}$'+r'}', 'tabTMScan',['S[table-format=1.0]', 'S[table-format=1.2]', 'S[table-format=2.2]', 'S[table-format=2.2]'], ["%1.0f", "%1.2f", "%2.2f", "%2.2f"])
dt_neu= np.delete(dt,24)
avg_t = avg_and_sem(dt_neu)
h = h/1000
avg_h = avg_and_sem(h)
print('Mittelwert von dt in s: ',avg_t)
print('Mittelwert von h in m: ',avg_h)
a=49.4/2000
V= avg_h[0]*math.pi/6*(3*a**2+avg_h[0]**2)
print('Kugelsegmentvolumen in m^3: ', V)
errV=(a**2/6+avg_h[0]**2*np.sqrt(3)/2)*math.pi*avg_h[1]
print('Fehler von V in m^3:',errV)
print('Relativer Fehler von V in %:', 100*(errV/V))


