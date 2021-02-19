from table2 import makeTable
from table2 import makeNewTable
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


#Erdmagnetfeld
def B_field(I,N,R):
	return const.mu_0 * 8/np.sqrt(125)* I*N/R

N_V = 20
R_V = 11.735*10**(-2) # m
I_V = 229*10**(-3) # A
print('vertical B field:', B_field(I_V,N_V,R_V))

v, I_S1, I_S2, I_H1, I_H2 = np.genfromtxt('scripts/horizontal.txt',unpack=True) # kHz,mA
v = v*1000 # Hz
I_S1 = (I_S1+12)*10**(-3) # A
I_S2 = (I_S2+12)*10**(-3) # A
I_H1 = I_H1*10**(-3) # A
I_H2 = I_H2*10**(-3) # A
R_S = 16.39*10**(-2) # m
N_S = 11
R_H = 15.79*10**(-2) # m
N_H = 154

B_S1 = B_field(I_S1,N_S,R_S)
B_S2 = B_field(I_S2,N_S,R_S)
B_H1 = B_field(I_H1,N_H,R_H)
B_H2 = B_field(I_H2,N_H,R_H)

B_1 = B_H1+B_S1
B_2 = B_H2+B_S2

makeTable([v/1000,I_S1*1000,I_H1*1000,B_S1*10**6,B_H1*10**6,B_1*10**6], r'{$\nu/\si{\kilo\hertz}$} & {$I_\text{S,A}/\si{\milli\ampere}$} & {$I_\text{H,A}/\si{\milli\ampere}$} & {$B_\text{S,A}/\si{\micro\tesla}$} & {$B_\text{H,A}/\si{\micro\tesla}$} & {$B_\text{Ges,A}/\si{\micro\tesla}$}','messung1A', ['S[table-format=4.0]','S[table-format=3.0]','S[table-format=3.0]','S[table-format=3.2]','S[table-format=3.2]','S[table-format=3.2]'], ["%4.0f", "%3.0f", "%3.0f", "%3.2f", "%3.2f", "%3.2f"])
makeTable([v/1000,I_S2*1000,I_H2*1000,B_S2*10**6,B_H2*10**6,B_2*10**6], r'{$\nu/\si{\kilo\hertz}$} & {$I_\text{S,B}/\si{\milli\ampere}$} & {$I_\text{H,B}/\si{\milli\ampere}$} & {$B_\text{S,B}/\si{\micro\tesla}$} & {$B_\text{H,B}/\si{\micro\tesla}$} & {$B_\text{Ges,B}/\si{\micro\tesla}$}','messung1B', ['S[table-format=4.0]','S[table-format=3.0]','S[table-format=3.0]','S[table-format=3.2]','S[table-format=3.2]','S[table-format=3.2]'], ["%4.0f", "%3.0f", "%3.0f", "%3.2f", "%3.2f", "%3.2f"])

def Linear(x,a,b):
	return a*x+b

params1, covariance_matrix1 = curve_fit(Linear,v,B_1)
errors1 = np.sqrt(np.diag(covariance_matrix1))
params2, covariance_matrix2 = curve_fit(Linear,v,B_2)
errors2 = np.sqrt(np.diag(covariance_matrix2))

a1 = unp.uarray(params1[0],errors1[0])
a2 = unp.uarray(params2[0],errors2[0])
print('a1 =', a1)
print('b1 =', unp.uarray(params1[1],errors1[1]))
print('a2 =', a2)
print('b2 =', unp.uarray(params2[1],errors2[1]))

x = np.linspace(0,1100)
plt.cla()
plt.clf()
plt.plot(v/1000,B_1*10**6,'rx',label=r'$B_\text{Ges,A}$')
plt.plot(x,Linear(x,params1[0]*1000*10**6,params1[1]*10**6),'b-',label=r'Ausgleichsgerade1')
plt.plot(v/1000,B_2*10**6,'mx',label=r'$B_\text{Ges,B}$')
plt.plot(x,Linear(x,params2[0]*1000*10**6,params2[1]*10**6),'c-',label=r'Ausgleichsgerade2')
plt.xlabel(r'$\nu/\si{\kilo\hertz}$')
plt.ylabel(r'$B/\si{\micro\tesla}$')
plt.xlim(0,1100)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('build/messung1.pdf')

print('horizontal B field:', avg_and_sem([params1[1],params2[1]]))


#Lande faktoren
def Lande(a):
	return 4*const.pi*const.m_e/(const.e*a)
def Kernspin(g_F):
	return 1/2*(2.0023/g_F-1)
	
g_F1 = Lande(a1)
g_F2 = Lande(a2)
print('g_F1:', g_F1)
print('g_F2:', g_F2)
I1 = Kernspin(g_F1)
I2 = Kernspin(g_F2)
print('I1:', I1)
print('I2:', I2)

#Zeeman_quad
def Zeeman_ges(g_F,B,DE,M_F):
	return g_F*const.physical_constants["Bohr magneton"][0]*B+(g_F*B*const.physical_constants["Bohr magneton"][0])**2*(1-2*M_F)/DE

def Zeeman_quad(g_F,B,DE,M_F):
	return (g_F*B*const.physical_constants["Bohr magneton"][0])**2*(1-2*M_F)/DE

print('DE_G1:', Zeeman_ges(g_F1,B_1[-1],4.53*10**(-24),1))
print('DE_G2:', Zeeman_ges(g_F2,B_2[-1],2.01*10**(-24),1))

print('DE_q1:', Zeeman_quad(g_F1,B_1[-1],4.53*10**(-24),1))
print('DE_q2:', Zeeman_quad(g_F2,B_2[-1],2.01*10**(-24),1))


#exponential fit
def exp_fkt(x,a,b):
	return (1-np.exp(-a*(x-b)))
	
t,t,t,x0,y0,t = np.genfromtxt('scripts/TEK0000.CSV',unpack=True,delimiter=",") 
y0 = np.array(y0[x0>=0])
for i in range(len(y0)-1):
	y0[i+1] = y0[i+1]-y0[0]
y0[0] = 0
y0 = y0/y0[-1]
x0 = np.array(x0[x0>=0]*1000)
t,t,t,x1,y1,t = np.genfromtxt('scripts/TEK0002.CSV',unpack=True,delimiter=",") 
y1 = np.array(y1[x1>=0])
for i in range(len(y1)-1):
	y1[i+1] = y1[i+1]-y1[0]
y1[0] = 0
y1 = y1/y1[-1]
x1 = np.array(x1[x1>=0]*1000)
t,t,t,x2,y2,t = np.genfromtxt('scripts/TEK0003.CSV',unpack=True,delimiter=",") 
y2 = y2[x2>=0]
x2 = x2[x2>=0]
y2 = y2[x2<=0.1]
x2 = x2[x2<=0.1]*1000
for i in range(len(y2)-1):
	y2[i+1] = y2[i+1]-y2[0]
y2[0] = 0
y2 = y2/y2[-1]

params0, covariance_matrix0 = curve_fit(exp_fkt,x0,y0,p0=(0.1,0))
errors0 = np.sqrt(np.diag(covariance_matrix0))
print('a0 =', unp.uarray(params0[0],errors0[0]))
print('b0 =', unp.uarray(params0[1],errors0[1]))
params1, covariance_matrix1 = curve_fit(exp_fkt,x1,y1,p0=(0.1,0))
errors1 = np.sqrt(np.diag(covariance_matrix1))
print('a1 =', unp.uarray(params1[0],errors1[0]))
print('b1 =', unp.uarray(params1[1],errors1[1]))
params2, covariance_matrix2 = curve_fit(exp_fkt,x2,y2,p0=(0.1,0))
errors2 = np.sqrt(np.diag(covariance_matrix2))
print('a2 =', unp.uarray(params2[0],errors2[0]))
print('b2 =', unp.uarray(params2[1],errors2[1]))

x = np.linspace(-1,105,1000)
plt.cla()
plt.clf()
plt.plot(x0,y0,'r-',label=r'Messwerte')
plt.plot(x,(exp_fkt(x,*params0)),'b-',label=r'Ausgleichskurve')
plt.xlabel(r'$t/\si{\milli\second}$')
plt.ylabel(r'Transparenz')
plt.xlim(-1,80)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('build/image0.pdf')

plt.cla()
plt.clf()
plt.plot(x1,y1,'r-',label=r'Messwerte')
plt.plot(x,(exp_fkt(x,*params1)),'b-',label=r'Ausgleichskurve')
plt.xlabel(r'$t/\si{\milli\second}$')
plt.ylabel(r'Transparenz')
plt.xlim(-1,60)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('build/image1.pdf')

plt.cla()
plt.clf()
plt.plot(x2,y2,'r-',label=r'Messwerte')
plt.plot(x,(exp_fkt(x,*params2)),'b-',label=r'Ausgleichskurve')
plt.xlabel(r'$t/\si{\milli\second}$')
plt.ylabel(r'Transparenz')
plt.xlim(-1,100)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('build/image2.pdf')


#Oszillationen
print('Oszillationen')
def hyper(x,a,b,c):
	return a+b/(x-c)
	
v, T1, T2 = np.genfromtxt('scripts/oszillationen.txt',unpack=True) # V,micro s
T1 = T1*10**(-3) #ms
T2 = T2*10**(-3) #ms

params1, covariance_matrix1 = curve_fit(hyper,v,T1)
errors1 = np.sqrt(np.diag(covariance_matrix1))
print('a1 =', unp.uarray(params1[0],errors1[0]))
b1 = unp.uarray(params1[1],errors1[1])
print('b1 =', b1)
print('c1 =', unp.uarray(params1[2],errors1[2]))
params2, covariance_matrix2 = curve_fit(hyper,v,T2)
errors2 = np.sqrt(np.diag(covariance_matrix2))
print('a2 =', unp.uarray(params2[0],errors2[0]))
b2 = unp.uarray(params2[1],errors2[1])
print('b2 =', b2)
print('c2 =', unp.uarray(params2[2],errors2[2]))

x = np.linspace(1,11,1000)
plt.cla()
plt.clf()
plt.plot(v,T1,'rx',label=r'Messwerte Resonanz1')
plt.plot(x,(hyper(x,*params1)),'b-',label=r'Ausgleichskurve')
plt.plot(v,T2,'mx',label=r'Messwerte Resonanz2')
plt.plot(x,(hyper(x,*params2)),'c-',label=r'Ausgleichskurve2')
plt.xlabel(r'$U/\si{\volt}$')
plt.ylabel(r'$T/\si{\milli\second}$')
plt.xlim(1,11)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('build/oszillationen.pdf')

print('Verhältnis:', b1/b2)
makeTable([v,T1,T2], r'{$U/\si{\volt}$} & {$T_1/\si{\milli\second}$} & {$T_2/\si{\milli\second}$} ','oszillationen', ['S[table-format=1.0]','S[table-format=1.2]','S[table-format=1.2]'], ["%1.0f", "%1.2f", "%1.2f"])









