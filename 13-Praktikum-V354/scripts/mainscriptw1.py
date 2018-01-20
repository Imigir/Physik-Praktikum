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

#5
def Dreieck(x,a):
	f=np.empty(len(x))
	x=x%(a)
	for i in range(0,len(x)):
		if x[i]<=a/2:
			f[i] = x[i]-a/4
		if a/2<x[i]:
			f[i] = -x[i]+3/4*a
	return f

def DreieckInt(x,a):
	F=np.empty(len(x))
	x=x%(a)
	for i in range(0,len(x)):
		if x[i]<=a/2:
			F[i] = 1/2*x[i]**2-a/4*x[i]
		if a/2<x[i]:
			F[i] = -1/2*x[i]**2+3/4*a*x[i]-1/4*a**2
	return F

plt.cla()
plt.clf()
x_plot = np.linspace(0,32,1000)
plt.plot(x_plot, 100*Dreieck(x_plot,6), 'b-', label='f(x)')
plt.plot(x_plot, 100*DreieckInt(x_plot,6), 'r-', label='F(x)')
plt.xlim(1,16)
plt.ylim(-200,200)
plt.xlabel(r'$t/\si{\micro\second}$')
plt.ylabel(r'$U/\si{\volt}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(framealpha=1, frameon=True)
plt.savefig("content/images/Graph5")

#6
plt.cla()
plt.clf()
x_plot = np.linspace(-2,20,1000)
plt.plot(x_plot, 100*np.sin(x_plot), 'b-', label='f(x)')
plt.plot(x_plot, 100*(np.cos(x_plot)+0.2), 'r-', label='F(x)')
plt.xlim(-pi/2,9/2*pi)
plt.ylim(-180,180)
#plt.xticks( [0, pi, 2*pi, 3*pi, 4*pi],[r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
plt.xlabel(r'$t/\si{\micro\second}$')
plt.ylabel(r'$U/\si{\volt}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(framealpha=1, frameon=True)
plt.savefig("content/images/Graph6")

#7
def Rechteck(x,a):
	f=np.empty(len(x))
	x=x%(a)
	c=1
	for i in range(0,len(x)):
		if x[i]<=a/2:
			f[i] = c
		if a/2<x[i]:
			f[i] = -c
	return f

def RechteckInt(x,a):
	F=np.empty(len(x))
	x=x%(a)
	c=1
	for i in range(0,len(x)):
		if x[i]<=a/2:
			F[i] = c*x[i]-a/4
		if a/2<x[i]:
			F[i] = -c*x[i]+3*a/4
	return F

plt.cla()
plt.clf()
x_plot = np.linspace(0,32,1000)
plt.plot(x_plot, 100*Rechteck(x_plot,3), 'b-', label='f(x)')
plt.plot(x_plot, 100*RechteckInt(x_plot,3), 'r-', label='F(x)')
plt.xlim(2.5,10)
plt.ylim(-180,180)
plt.xlabel(r'$t/\si{\micro\second}$')
plt.ylabel(r'$U/\si{\volt}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(framealpha=1, frameon=True)
plt.savefig("content/images/Graph7")
