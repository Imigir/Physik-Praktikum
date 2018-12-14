from table2 import makeTable
from table2 import makeNewTable
from linregress import linregress
from customFormatting import *
from bereich import bereich
from weightedavgandsem import weighted_avg_and_sem
from weightedavgandsem import avg_and_sem
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit as cf
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
import uncertainties
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

def Plot(Werte, name, funktionParams=(1,0), xname='$T$'):
	plt.cla()
	plt.clf()
	plt.plot(Werte[0], Werte[1], 'rx', label='Wertepaare')
	plt.xlabel(xname)
	plt.ylabel(r'$I$')
	#plt.yscale('log')
	plt.legend(loc='best')
	plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
	#plt.savefig('content/images/'+name+'.png')

def expFunktion(x, a, b, c):
	return c+a*np.exp(b*x)

def gaus(x, a, c,sigma,b):
	return a*np.exp(-(x-b)**2/(2*sigma**2))+c

t,T,I = np.genfromtxt('scripts/data1.txt',unpack=True)
print('Plot1')
#Plot([T,I],'Plot1')

Te, Ie =np.genfromtxt('scripts/datafit1.txt',unpack=True)

params, covar = cf(expFunktion, Te, Ie, maxfev=10000)
#paramsEQU=uncertainties.correlated_values(params, covar)
#print(I)
for i in range(len(I)):
	I[i]=I[i]-expFunktion(T[i], *params)
#print(I)
#print(paramsEQU)

T=T+273.15

#params1,covar1=cf(gaus, T[0:32], I[0:32], p0=[10,1,10,2.5*10**2], maxfev=10000)
#a1=unp.uarray(params1[0],np.sqrt(covar1[0][0]))
#c1=unp.uarray(params1[1],np.sqrt(covar1[1][1]))
#sigma1=unp.uarray(params1[2],np.sqrt(covar1[2][2]))
#b1=unp.uarray(params1[3],np.sqrt(covar1[3][3]))
xplot=np.linspace(-50,15,1000)
plt.cla()
plt.clf()
#plt.plot(xplot+273.15,gaus(xplot+273.15,*params1))
plt.plot(T[0:32], I[0:32], 'rx', label='Wertepaare')
plt.xlabel(r'$T$')
plt.ylabel(r'$I$')
#plt.yscale('log')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.show()


#print('1: ',a1,b1,c1,sigma1)

#W: 1.Möglichkeit
def linear(x,a,b):
	return a*x+b
print(I[0:15])
#params1,covar1 = cf(linear,1/T[0:15],np.log(I[0:15]))
#a1=unp.uarray(params[0],np.sqrt(covar1[0][0]))
#b1=unp.uarray(params[1],np.sqrt(covar1[1][1]))
#print('a1: ', a1, ',b1: ',b1)
plt.cla()
plt.clf()
#plt.plot(1/(xplot+273.15),linear(1/(xplot+273.15),*params1),'b-', label='Ausgleichgerade')
plt.plot(1/T[0:15],np.log(I[0:15]), 'rx', label='Wertepaare')
plt.xlabel(r'$1/T$')
plt.ylabel(r'$ln(I)$')
#plt.yscale('log')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.show()


#W: 2.Möglichkeit
Fläche1=np.empty(32)
for i in range(32):
	Fläche1[i]=np.trapz(I[i:32],T[i:32])
print('Fläche des 1.Plots: ',Fläche1[0],'pA K')

x=1/T[0:32]

plt.cla()
plt.clf()
plt.plot(x, np.log(Fläche1/285.65/I[0:32]), 'b.', label='Wertepaare')
plt.xlabel(r'$1/T$')
plt.ylabel(r'$I/iT_.max$')
#plt.yscale('log')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.show()


'''
t2,T2,I2 = np.genfromtxt('scripts/data2.txt',unpack=True)

Te2, Ie2 =np.genfromtxt('scripts/datafit2.txt',unpack=True)

params2, covar2 = cf(expFunktion, Te2, Ie2, maxfev=10000)
#paramsEQU=uncertainties.correlated_values(params, covar)
print(I2)
for i in range(len(I2)):
	I2[i]=I2[i]-expFunktion(T2[i], *params2)
print(I2)
#print(paramsEQU)


params2,covar2=cf(gaus, T2[8:30], I2[8:30], maxfev=10000)
a2=unp.uarray(params2[0],np.sqrt(covar2[0][0]))
c2=unp.uarray(params2[1],np.sqrt(covar2[1][1]))
sigma2=unp.uarray(params2[2],np.sqrt(covar2[2][2]))
b2=unp.uarray(params2[3],np.sqrt(covar2[3][3]))

plt.cla()
plt.clf()
plt.plot(xplot,gaus(xplot,*params2))
plt.plot(T2[8:30], I2[8:30], 'rx', label='Wertepaare')
plt.xlabel(r'$T$')
plt.ylabel(r'$I$')
#plt.yscale('log')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.show()

Fläche2=np.trapz(I2[8:30],T2[8:30])
print('Fläche des 2.Plots: ',Fläche2,'pA K')
print('Plot2')
Plot([T,I],'Plot2')
'''
