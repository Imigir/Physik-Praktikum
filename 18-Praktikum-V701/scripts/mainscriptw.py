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
from scipy.special import factorial
from scipy.stats import poisson
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
import scipy.constants as const

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
plt.xlim(0, t[-1]*100)
plt.xlabel(r'$v/\si{\centi\meter\per\second}$')
plt.ylabel(r'$\Delta f / \si{\hertz}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/Graph.pdf')

a = unp.uarray(params[0], np.sqrt(covar[0][0]))
params = unp.uarray(params, np.sqrt(np.diag(covar)))
makeNewTable([convert((r'$c_\text{1}$',r'$c_\text{2}$',r'$T_{\text{A}1}$',r'$T_{\text{A}2}$',r'$\alpha$',r'$D_1$',r'$D_2$',r'$A_1$',r'$A_2$',r'$A_3$',r'$A_4$'),strFormat),convert(np.array([paramsGes2[0],paramsGes1[0],deltat2*10**6,deltat1*10**6,-paramsDaempfung[0]*2,4.48*10**-6 *paramsGes1[0]/2*10**3, 7.26*10**-6 *paramsGes1[0]/2*10**3, (VierteMessung-2*deltat2*10**6)[0]*10**-6 *1410 /2*10**3, unp.uarray((VierteMessung[1]-VierteMessung[0])*10**-6 *1410 /2*10**3, 0), unp.uarray((VierteMessung[2]-VierteMessung[1])*10**-6 *2500 /2*10**3, 0),unp.uarray((VierteMessung[3]-VierteMessung[2])*10**-6 *1410 /2*10**3, 0)]),unpFormat,[[r'\meter\per\second',"",True],[r'\meter\per\second',"",True],[r'\micro\second',"",True],[r'\micro\second',"",True],[r'\per\meter',"",True],[r'\milli\meter',"",True],[r'\milli\meter',"",True],[r'\milli\meter',"",True],[r'\milli\meter',r'1.3f',True],[r'\milli\meter',r'1.3f',True],[r'\milli\meter',r'2.2f',True]]),convert(np.array([2730,2730]),floatFormat,[r'\meter\per\second','1.0f',True])+convert((r'-',r'-'),strFormat)+convert(unp.uarray([57,6.05,9.9],[2.5,0,0]),unpFormat,[[r'\per\meter',"",True],[r'\milli\meter',r'1.2f',True],[r'\milli\meter',r'1.2f',True]])+convert((r'-',r'-',r'-',r'-'),strFormat),convert(np.array([(2730-paramsGes2[0])/2730*100,(2730-paramsGes1[0])/2730*100]),unpFormat,[r'\percent','',True])+convert((r'-',r'-'),strFormat)+convert(np.array([(-paramsDaempfung[0]*2-unp.uarray(57,2.5))/unp.uarray(57,2.5)*100,(4.48*10**-6 *paramsGes1[0]/2*10**3-6.05)/6.05*100, (-7.26*10**-6 *paramsGes1[0]/2*10**3+9.90)/9.90*100]),unpFormat,[r'\percent','',True])+convert((r'-',r'-',r'-',r'-'),strFormat)],r'{Wert}&{gemessen}&{Literaturwert\cite{cAcryl},\cite{alphaAcryl}}&{Abweichung}','Ergebnisse', ['c ','c',r'c','c'])
"""

p0 = 1.013

def xeff(p, x0):
	return x0*p/p0
	
def G(x, m, s):
	return 1/np.sqrt(2*np.pi*s**2)*np.exp(-(x-m)**2/(2*s**2))
	
def P(x, m):
	return m**(x)/factorial(x)*np.exp(-m)
	

#1)
p, N1, c1 = np.genfromtxt('scripts/data1.txt',unpack=True)
p = p/1000
N1 = N1/60
E1 = 4*10**6/c1[0]*c1[:17]
x1 = 2.7/100

paramsLinear2, errorsLinear2, sigma_y = linregress(xeff(p,x1)[13:18], N1[13:18])
steigung2 = unp.uarray(paramsLinear2[0], errorsLinear2[0])
achsenAbschnitt2 = unp.uarray(paramsLinear2[1], errorsLinear2[1])

Rm1 = (N1[0]/2-achsenAbschnitt2)/steigung2
Ea1 = (Rm1*1000/3.1)**(1./3)*10**6

print('1)')
print('xeff1=',xeff(p,x1)[13])
print('xeff2=',xeff(p,x1)[17])
print('N1/2=',N1[0]/2)
print('steigung2=', steigung2)
print('achsenabschhnitt2=', achsenAbschnitt2)
print('Rm1 in m=', Rm1)
print('Ea1 in eV=', Ea1)
"""
plt.cla()
plt.clf()
x_plot = np.linspace(-5,30)
plt.plot(xeff(p,x1*1000), N1, 'rx', label='Messwerte')
plt.plot(x_plot, x_plot/1000*paramsLinear2[0]+paramsLinear2[1], 'b-', label='Ausgleichsgerade')
plt.xlim(-1, 28)
plt.ylim(-100, 1200)
plt.xlabel(r'$x_\text{eff}/\si{\milli\meter}$')
plt.ylabel(r'$N/\si{\becquerel}$')
plt.legend(loc='best')
plt.savefig('content/images/Graph1N.pdf')
"""
paramsLinear, errorsLinear, sigma_y = linregress(xeff(p,x1)[:17], E1)
steigung = unp.uarray(paramsLinear[0], errorsLinear[0])
achsenAbschnitt = unp.uarray(paramsLinear[1], errorsLinear[1])
"""
plt.cla()
plt.clf()
x_plot = np.linspace(-1,22)
plt.plot(xeff(p,x1*1000)[:17], E1/10**6, 'rx', label='Messwerte')
plt.plot(x_plot, (x_plot/1000*paramsLinear[0]+paramsLinear[1])/10**6, 'b-', label='Ausgleichsgerade')
plt.xlim(-1, 22)
plt.xlabel(r'$x_\text{eff}/\si{\milli\meter}$')
plt.ylabel(r'$E/\si{\mega e\volt}$')
plt.legend(loc='best')
plt.savefig('content/images/Graph1E.pdf')
"""

print('steigung=', steigung)
print('E0=', achsenAbschnitt)
"""
E1 = np.append(E1, [-1, -1, -1])
makeTable([p*1000, N1, xeff(p,x1*1000), E1/10**6], r'{'+r'$p/\si{\milli\bar}$'+r'} & {'+r'$N_.1/\si{\becquerel}$'+r'} & {'+r'$x_.{eff1}/\si{\milli\metre}$'+r'} & {'+r'$E_1/\si{\mega e\volt}$'+r'}', 'tab1', ['S[table-format=3.0]', 'S[table-format=3.0]', 'S[table-format=2.1]', 'S[table-format=1.2]'], ["%3.0f", "%3.0f", "%2.1f", "%1.2f"])
"""
#2)
p, N2, c2 = np.genfromtxt('scripts/data2.txt',unpack=True)
p = p/1000
N2 = N2/60
E2 = 4*10**6/c2[0]*c2
x2 = 1/100
"""
plt.cla()
plt.clf()
plt.plot(xeff(p,x2*1000), N2, 'rx', label='Messwerte')
#plt.plot(x, y, 'b-', label='Ausgleichsgerade')
#plt.xlim(0, 100)
plt.ylim(0, 3300)
plt.xlabel(r'$x_\text{eff}/\si{\milli\meter}$')
plt.ylabel(r'$N/\si{\becquerel}$')
plt.legend(loc='best')
plt.savefig('content/images/Graph2N.pdf')
"""
paramsLinear, errorsLinear, sigma_y = linregress(xeff(p,x2), E2)
steigung = unp.uarray(paramsLinear[0], errorsLinear[0])
achsenAbschnitt = unp.uarray(paramsLinear[1], errorsLinear[1])
"""
plt.cla()
plt.clf()
x_plot = np.linspace(-1,11)
plt.plot(xeff(p,x2*1000), E2/10**6, 'rx', label='Messwerte')
plt.plot(x_plot, (x_plot/1000*paramsLinear[0]+paramsLinear[1])/10**6, 'b-', label='Ausgleichsgerade')
plt.xlim(-1, 11)
plt.xlabel(r'$x_\text{eff}/\si{\milli\meter}$')
plt.ylabel(r'$E/\si{\mega e\volt}$')
plt.legend(loc='best')
plt.savefig('content/images/Graph2E.pdf')
"""
print('2)')
print('steigung=', steigung)
print('E0=', achsenAbschnitt)

makeTable([p*1000, N2, xeff(p,x2*1000), E2/10**6], r'{'+r'$p/\si{\milli\bar}$'+r'} & {'+r'$N_.2/\si{\becquerel}$'+r'} & {'+r'$x_.{eff2}/\si{\milli\metre}$'+r'} & {'+r'$E_2/\si{\mega e\volt}$'+r'}', 'tab2', ['S[table-format=3.0]', 'S[table-format=3.0]', 'S[table-format=2.1]', 'S[table-format=1.2]'], ["%3.0f", "%3.0f", "%2.1f", "%1.2f"])

#3)
N3 = np.genfromtxt('scripts/data3.txt',unpack=True)
N3 = N3/10
#N3 = np.rint(N3)
N3.sort()
mu, sigma = avg_and_sem(N3)
sigma = sigma*np.sqrt(len(N3))
#placeholder = N3+500
#x = mu + sigma * np.random.randn(10000)

print('3)')
print('Mittelwert=', mu)
print('Standardabweichung=', sigma)

"""
x = np.linspace(600,740, 141)
#x = np.linspace(0,14, 15)
y = poisson.pmf(x, mu)*10000
#y = P(x, mu/10-60)*10000
y = np.rint(y)
print('y', y)
for i in range(0, len(y)):
	for j in range(1,int(y[i])):
		x = np.append(x,x[i])
#x = (x+60)*10

plt.cla()
plt.clf()
x_plot = np.linspace(600,740, 1000)
plt.hist([N3, x], bins=np.linspace(600, 740, 8), range=(600, 740), normed=True, label=['Messdaten', 'Poissonverteilung'])
plt.plot(x_plot, G(x_plot, mu, sigma), label='Gaußverteilung')
plt.xlim(600, 740)
plt.xlabel(r'$N/\si{\becquerel}$')
plt.ylabel('Häufigkeit')
plt.legend(loc='best')
plt.savefig('content/images/Graph3_20.pdf')

print('Graph3_20 finished')

plt.cla()
plt.clf()
x_plot = np.linspace(600,740, 1000)
plt.hist([N3, x], bins=np.linspace(600, 740, 15), range=(600, 740), normed=True, label=['Messdaten', 'Poissonverteilung'])
plt.plot(x_plot, G(x_plot, mu, sigma), label='Gaußverteilung')
plt.xlim(600, 740)
plt.xlabel(r'$N/\si{\becquerel}$')
plt.ylabel('Häufigkeit')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('content/images/Graph3_10.pdf')

print('Graph3_10 finished')

plt.cla()
plt.clf()
x_plot = np.linspace(600,740, 1000)
plt.hist([N3, x], bins='auto', range=(600, 740), normed=True, label=['Messdaten', 'Poissonverteilung'])
plt.plot(x_plot, G(x_plot, mu, sigma), label='Gaußverteilung')
plt.xlim(600, 740)
plt.xlabel(r'$N/\si{\becquerel}$')
plt.ylabel('Häufigkeit')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('content/images/Graph3_auto.pdf')

print('Graph3_auto finished')

plt.cla()
plt.clf()
x_plot = np.linspace(600,740, 1000)
plt.hist([N3, x], bins='fd', range=(600, 740), normed=True, label=['Messdaten', 'Poissonverteilung'])
plt.plot(x_plot, G(x_plot, mu, sigma), label='Gaußverteilung')
plt.xlim(600, 740)
plt.xlabel(r'$N/\si{\becquerel}$')
plt.ylabel('Häufigkeit')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('content/images/Graph3_fd.pdf')

print('Graph3_fd finished')

plt.cla()
plt.clf()
x_plot = np.linspace(600,740, 1000)
plt.hist([N3, x], bins='doane', range=(600, 740), normed=True, label=['Messdaten', 'Poissonverteilung'])
plt.plot(x_plot, G(x_plot, mu, sigma), label='Gaußverteilung')
plt.xlim(600, 740)
plt.xlabel(r'$N/\si{\becquerel}$')
plt.ylabel('Häufigkeit')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('content/images/Graph3_doane.pdf')

print('Graph3_doane finished')

plt.cla()
plt.clf()
x_plot = np.linspace(600,740, 1000)
plt.hist([N3, x], bins='scott', range=(600, 740), normed=True, label=['Messdaten', 'Poissonverteilung'])
plt.plot(x_plot, G(x_plot, mu, sigma), label='Gaußverteilung')
plt.xlim(600, 740)
plt.xlabel(r'$N/\si{\becquerel}$')
plt.ylabel('Häufigkeit')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('content/images/Graph3_scott.pdf')

print('Graph3_scott finished')

plt.cla()
plt.clf()
x_plot = np.linspace(600,740, 1000)
plt.hist([N3, x], bins='rice', range=(600, 740), normed=True, label=['Messdaten', 'Poissonverteilung'])
plt.plot(x_plot, G(x_plot, mu, sigma), label='Gaußverteilung')
plt.xlim(600, 740)
plt.xlabel(r'$N/\si{\becquerel}$')
plt.ylabel('Häufigkeit')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('content/images/Graph3_rice.pdf')

print('Graph3_rice finished')
"""

"""
plt.cla()
plt.clf()
x_plot = np.linspace(600,740, 1000)
plt.hist([N3, x], bins='sturges', range=(600, 740), normed=True, label=['Messdaten', 'Poissonverteilung'])
plt.plot(x_plot, G(x_plot, mu, sigma), label='Gaußverteilung')
plt.xlim(600, 740)
plt.xlabel(r'$N/\si{\becquerel}$')
plt.ylabel('Häufigkeit')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('content/images/Graph3_sturges.pdf')

print('Graph3_sturges finished')

plt.cla()
plt.clf()
x_plot = np.linspace(600,740, 1000)
plt.hist([N3, x], bins='sqrt', range=(600, 740), normed=True, label=['Messdaten', 'Poissonverteilung'])
plt.plot(x_plot, G(x_plot, mu, sigma), label='Gaußverteilung')
plt.xlim(600, 740)
plt.xlabel(r'$N/\si{\becquerel}$')
plt.ylabel('Häufigkeit')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('content/images/Graph3_sqrt.pdf')

print('Graph3_sqrt finished')
"""
