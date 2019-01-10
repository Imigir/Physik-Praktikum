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

def Plot(Werte, name, xname=r'$T/\si{\kelvin}$', yname=r'$I/\si{\pico\ampere}$'):
	plt.cla()
	plt.clf()
	plt.plot(Werte[0], Werte[1], 'r.', label='Wertepaare')
	plt.xlabel(xname)
	plt.ylabel(yname)
	plt.legend(loc='best')
	plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
	plt.savefig('content/images/'+name+'.pdf')

def expFunktion(x, a, b):
	return np.exp(a*(x-b))

def gaus(x, a, c,sigma,b):
	return a*np.exp(-(x-b)**2/(2*sigma**2))+c

def linear(x,a,b):
	return a*x+b

t,T,I = np.genfromtxt('scripts/data1.txt',unpack=True)
T=T+273.15
print('Plot1')
Plot([T,I],'Plot1')

xplot=np.linspace(-50,50,1000)

Te, Ie =np.genfromtxt('scripts/datafit1_2.txt',unpack=True)
Te=Te+273.15

#params, covar = cf(expFunktion, Te, Ie,p0=[1,0.05,0.5], maxfev=10000)
params, covar = cf(expFunktion, Te, Ie, maxfev=10000)

plt.cla()
plt.clf()
plt.plot(T,I,'y.',label='Messwerte')
plt.plot(Te,Ie,'r.',label='verwendete Werte')
plt.plot(xplot+273.15,expFunktion(xplot+273.15,*params),'b-',label='Ausgleichskurve')
plt.xlabel(r'$T/\si{\kelvin}$')
plt.ylabel(r'$i/\si{\pico\ampere}$')
plt.ylim(-10,50)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/plot1exp.pdf')

paramsU=uncertainties.correlated_values(params, covar)
print(paramsU)
I_roh=I
for i in range(len(I)):
	I[i]=I[i]-expFunktion(T[i], *params)
makeTable([t,T,I_roh,I], r'{'+r'$t_\text{1}/(\si{\minute})$'+r'} & {'+r'$T_\text{1}/(\si{\kelvin})$'+r'} & {'+r'$I_\text{roh,1}/(\si{\pico\ampere})$'+r'} & {'+r'$I_\text{ber,1}/(\si{\pico\ampere})$'+r'}','tabData1',['S[table-format=2.0]','S[table-format=3.1]','S[table-format=2.1]','S[table-format=2.1]'],["%2.0f","%3.1f","%2.1f","%2.1f"])

#W: 1.Möglichkeit
print('erste Möglichkeit')

x=1/T
params4,covar4=cf(linear,x[4:15],np.log(I[4:15]))

paramsU=uncertainties.correlated_values(params4, covar4)
print(paramsU)

plt.cla()
plt.clf()
plt.plot(x[:30],np.log(I[:30]),'y.',label='Messwerte')
plt.plot(x[4:15],np.log(I[4:15]),'r.',label='gefittete Messwerte')
plt.plot(1/(xplot+273.15), linear(1/(xplot+273.15),*params4),'b-',label='Ausgleichsgerade')
plt.xlabel(r'$T^{-1}/\si{\kelvin^{-1}}$')
plt.ylabel(r'$\ln(i/\si{\pico\ampere})$')
plt.xlim(0.0038,0.0046)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/W1_1.pdf')
W1_1=unp.uarray(params4[0],np.sqrt(covar4[0][0]))*(-const.k)


#params1,covar1=cf(gaus, T[0:31], I[0:31], p0=[10,1,10,2.5*10**2], maxfev=10000)
#a1=unp.uarray(params1[0],np.sqrt(covar1[0][0]))
#c1=unp.uarray(params1[1],np.sqrt(covar1[1][1]))
#sigma1=unp.uarray(params1[2],np.sqrt(covar1[2][2]))
#b1=unp.uarray(params1[3],np.sqrt(covar1[3][3]))
plt.cla()
plt.clf()
#plt.plot(xplot+273.15,gaus(xplot+273.15,*params1))
plt.plot(T[0:31], I[0:31], 'r.', label='Wertepaare')
plt.xlabel(r'$T/\si{\kelvin}$')
plt.ylabel(r'$i/\si{\pico\ampere}$')
#plt.yscale('log')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/bereinigt1.pdf')


#print('1: ',a1,b1,c1,sigma1)



#W: 2.Möglichkeit
print('zweite Möglichkeit')

Fläche1=np.empty(32)
for i in range(0,32):
	Fläche1[i]=np.trapz(I[i:32],T[i:32])

print('Fläche des 1.Plots: ',Fläche1[0],'pA K')

params3,covar3 =cf(linear, x[4:23], np.log(Fläche1[4:23]/I[4:23]),maxfev=10000)
#params3,covar3 =cf(linear, x[0:31], np.log(Fläche1[0:31]/I[0:31]),maxfev=10000)

paramsU=uncertainties.correlated_values(params3, covar3)
print(paramsU)

plt.cla()
plt.clf()
plt.plot(1/(xplot+273.15),linear(1/(xplot+273.15),*params3),'b-', label='Ausgleichsgerade')
plt.plot(x[0:31], np.log(Fläche1[0:31]/I[0:31]), 'y.', label='Wertepaare')
plt.plot(x[4:23], np.log(Fläche1[4:23]/I[4:23]), 'r.', label='gefittete Wertepaare')
plt.xlabel(r'$T^{-1}/\si{\kelvin}$')
plt.ylabel(r'$\frac{I}{i T_\text{max}}$')
plt.xlim(0.0034,0.0046)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/W1_2.pdf')

W1_2=unp.uarray(params3[0],np.sqrt(covar3[0][0]))*(const.k)
print('W1 1. Möglichkeit: ', W1_1,'eV: ', W1_1/const.e)
print('W1 2. Möglichkeit: ', W1_2,'eV: ', W1_2/const.e)


#Plot 2

t2,T2,I2 = np.genfromtxt('scripts/data2.txt',unpack=True)
T2=T2+273.15
print('Plot2')
Plot([T2,I2],'Plot2')

Te2, Ie2 =np.genfromtxt('scripts/datafit2_2.txt',unpack=True)
Te2=Te2+273.15

#params2_1, covar2_1 = cf(expFunktion, Te2, Ie2, p0=[1,0.05,0.5], maxfev=10000)
params2_1, covar2_1 = cf(expFunktion, Te2, Ie2, maxfev=10000)

plt.cla()
plt.clf()
plt.plot(T2,I2,'y.',label='Messwerte')
plt.plot(Te2,Ie2,'r.',label='gefittete Werte')
plt.plot(xplot+273.15,expFunktion(xplot+273.15,*params2_1),'b-',label='Ausgleichskurve')
plt.xlabel(r'$T/\si{kelvin}$')
plt.ylabel(r'$i/\si{\pico\ampere}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/plot2exp.pdf')

paramsU=uncertainties.correlated_values(params2_1, covar2_1)
print(paramsU)
I_roh2=I2
for i in range(len(I2)):
	I2[i]=I2[i]-expFunktion(T2[i], *params2_1)
makeTable([t2,T2,I_roh2,I2], r'{'+r'$t_\text{2}/(\si{\minute})$'+r'} & {'+r'$T_\text{2}/(\si{\kelvin})$'+r'} & {'+r'$I_\text{roh,2}/(\si{\pico\ampere})$'+r'} & {'+r'$I_\text{ber,2}/(\si{\pico\ampere})$'+r'}','tabData2',['S[table-format=2.0]','S[table-format=3.1]','S[table-format=2.1]','S[table-format=2.1]'],["%2.0f","%3.1f","%2.1f","%2.1f"])



#1.Möglichkeit
print('erstse Möglichkeit')

x2=1/T2
params2_2,covar2_2 = cf(linear,1/T2[0:15],np.log(I2[0:15]),maxfev=10000)

paramsU=uncertainties.correlated_values(params2_2, covar2_2)
print(paramsU)

a1=unp.uarray(params2_2[0],np.sqrt(covar2_2[0][0]))
b1=unp.uarray(params2_2[1],np.sqrt(covar2_2[1][1]))
#print('a1:', a1, ',b1:',b1)

plt.cla()
plt.clf()
plt.plot(1/(xplot+273.15),linear(1/(xplot+273.15),*params2_2),'b-', label='Ausgleichgerade')
plt.plot(x2[:30],np.log(I2[:30]), 'y.', label='Wertepaare')
plt.plot(x2[0:15],np.log(I2[0:15]), 'r.', label='gefittete Wertepaare')
plt.xlabel(r'$T^{-1}/\si{\kelvin^{-1}}$')
plt.ylabel(r'$\ln(i/\si{\pico\ampere})$')
plt.xlim(0.0038,0.0044)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/W2_1.pdf')
W2_1=a1*(-const.k)


#2.Möglichkeit
print('zweite Möglichkeit')

Fläche2=np.empty(35)
for i in range(0,35):
	Fläche2[i]=np.trapz(I2[i:35],T2[i:35])

print('Fläche des 2.Plots: ',Fläche2[0],'pA K')

params2_3,covar2_3 =cf(linear, 1/T2[0:26], np.log(Fläche2[0:26]/I2[0:26]),maxfev=10000)
#params2_3,covar2_3 =cf(linear, 1/T2[3:33], np.log(Fläche2[3:33]/I2[3:33]),maxfev=10000)

paramsU=uncertainties.correlated_values(params2_3, covar2_3)
print(paramsU)

plt.cla()
plt.clf()
plt.plot(1/(xplot+273.15),linear(1/(xplot+273.15),*params2_3),'b-', label='Ausgleichsgerade')
plt.plot(1/T2[0:34], np.log(Fläche2[0:34]/I2[0:34]), 'y.', label='Wertepaare')
plt.plot(1/T2[0:26], np.log(Fläche2[0:26]/I2[0:26]), 'r.', label='gefittete Wertepaare')
plt.xlabel(r'$T^{-1}/\si{\kelvin}$')
plt.ylabel(r'$\frac{I}{i T_\text{max}}$')
plt.xlim(0.0036,0.0044)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/W2_2.pdf')

W2_2=unp.uarray(params2_3[0],np.sqrt(covar2_3[0][0]))*(const.k)

print('W2 1. Möglichkeit: ', W2_1,'eV: ', W2_1/const.e)
print('W2 2. Möglichkeit: ', W2_2,'eV: ', W2_2/const.e)


plt.cla()
plt.clf()
#plt.plot(xplot,gaus(xplot,*params2))
plt.plot(T2[0:38], I2[0:38], 'r.', label='Wertepaare')
plt.xlabel(r'$T/\si{\kelvin}$')
plt.ylabel(r'$i/\si{\pico\ampere}$')
#plt.yscale('log')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/bereinigt2.pdf')

#tau_0
b1_array=np.empty(52)
for i in range(len(b1_array)-1):
	b1_array[i]=(T[i+1]-T[i])/(t[i+1]-t[i])

b2_array=np.empty(61)
for i in range(len(b2_array)-1):
	b2_array[i]=(T2[i+1]-T2[i])/(t2[i+1]-t2[i])

b1=avg_and_sem(b1_array)
b1=unp.uarray(b1[0],b1[1])
b2=avg_and_sem(b2_array)
b2=unp.uarray(b2[0],b2[1])

print('b1 =', b1)
print('b2 =', b2)

W1_avg=avg_and_sem([W1_1.n,W1_2.n])
W1_avg=unp.uarray(W1_avg[0],W1_avg[1])
W2_avg=avg_and_sem([W2_1.n,W2_2.n])
W2_avg=unp.uarray(W2_avg[0],W2_avg[1])
W_avg=avg_and_sem([W1_1.n,W1_2.n,W2_1.n,W2_2.n])
W_avg=unp.uarray(W_avg[0],W_avg[1])
print('W_avg =', W_avg, 'eV: ', W_avg/const.e)

Tmax=-12.5+273.15
c=(const.k*Tmax**2)
a=c/Tmax

tau_0_1=c/W_avg/b1*unp.exp(-W_avg/a)

#tau_0_1_1=c/W1_1/b1*unp.exp(-W1_1/a)
#tau_0_1_2=c/W1_2/b1*unp.exp(-W1_2/a)

#tau_fehler_1_1=c**2*np.exp(-2*W1_1.n/a)/(W1_1.n)**2/b1[0]**4*b1[1]**2+c**2*np.exp(-2*W1_1.n/a)*(const.k+W1_1.n)**2/a**2/(W1_1.n)**4/b1[0]**2*W1_1.s**2
#tau_fehler_1_1=np.sqrt(tau_fehler_1_1)
#tau_fehler_1_2=c**2*np.exp(-2*W1_2.n/a)/(W1_2.n)**2/b1[0]**4*b1[1]**2+c**2*np.exp(-2*W1_2.n/a)*(const.k+W1_2.n)**2/a**2/(W1_2.n)**4/b1[0]**2*W1_2.s**2
#tau_fehler_1_2=np.sqrt(tau_fehler_1_2)


Tmax=-13.4+273.15
c=(const.k*Tmax**2)
a=c/Tmax

tau_0_2=c/W_avg/b2*unp.exp(-W_avg/a)

#tau_0_2_1=c/W2_1/b2*unp.exp(-W2_1/a)
#tau_0_2_2=c/W2_2/b2*unp.exp(-W2_2/a)

#tau_fehler_2_1=c**2*np.exp(-2*W2_1.n/a)/(W2_1.n)**2/b2[0]**4*b1[1]**2+c**2*np.exp(-2*W2_1.n/a)*(const.k+W2_1.n)**2/a**2/(W2_1.n)**4/b2[0]**2*W2_1.s**2
#tau_fehler_2_1=np.sqrt(tau_fehler_2_1)
#tau_fehler_2_2=c**2*np.exp(-2*W2_2.n/a)/(W2_2.n)**2/b2[0]**4*b1[1]**2+c**2*np.exp(-2*W2_2.n/a)*(const.k+W2_2.n)**2/a**2/(W2_2.n)**4/b2[0]**2*W2_2.s**2
#tau_fehler_2_2=np.sqrt(tau_fehler_2_2)


#print(tau_0_1_1,'+-', tau_fehler_1_1)
#print(tau_0_1_2,'+-', tau_fehler_1_2)
#print(tau_0_2_1,'+-', tau_fehler_2_1)
#print(tau_0_2_2,'+-', tau_fehler_2_2)

print('tau:')
print('1: ',tau_0_1)
print('2: ',tau_0_2)
#print('1_1:', tau_0_1_1)
#print('1_2:', tau_0_1_2)
#print('2_1:', tau_0_2_1)
#print('2_2:', tau_0_2_2)
