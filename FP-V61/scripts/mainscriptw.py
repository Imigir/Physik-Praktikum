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


#Stabilitätsprüfung
def gi1(L,r1,r2):
	return (1-L/r1)*(1-L/r2)

def gi2(L,r):
	return (1-L/r)

x = np.linspace(0,280,10000)
y = np.linspace(0,140,5000)
plt.cla()
plt.clf()
plt.plot(y, gi2(y,140),'k--',label='Stabilitätskurve für (planar, konkav: r=140 cm)')
plt.plot(x, gi1(x,140,140), 'r', label='Stabilitätskurve für (konkav: r=140 cm, konkav: r=140 cm)')
plt.xlabel(r'$L/\si{\centi\meter}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.04, w_pad=1.08)
plt.savefig('build/'+'stabilitat')


#longitudinale Moden

def lambda_fkt(L,a):
	return a*L

f_192  = np.genfromtxt('scripts/data1.txt',unpack=True) #MHz
f_192 = f_192*10**6 #Hz
lambda_192 = f_192[:len(f_192)-1]
for i in range(len(f_192)-1):
	lambda_192[i] = const.c/(f_192[i+1]-f_192[i])
lambda_192 = avg_and_sem(lambda_192)
print('lambda_192:', lambda_192)

f_120  = np.genfromtxt('scripts/data2.txt',unpack=True) #MHz
f_120 = f_120*10**6 #Hz
lambda_120 = f_120[:len(f_120)-1]
for i in range(len(f_120)-1):
	lambda_120[i] = const.c/(f_120[i+1]-f_120[i])
lambda_120 = avg_and_sem(lambda_120)
print('lambda_120:', lambda_120)

f_71  = np.genfromtxt('scripts/data3.txt',unpack=True) #MHz
f_71 = f_71*10**6 #Hz
lambda_71 = f_71[:len(f_71)-1]
for i in range(len(f_71)-1):
	lambda_71[i] = const.c/(f_71[i+1]-f_71[i])
lambda_71 = avg_and_sem(lambda_71)
print('lambda_71:', lambda_71)

lambda_ges=unp.uarray([lambda_71[0],lambda_120[0],lambda_192[0]],[lambda_71[1],lambda_120[1],lambda_192[1]])
print('lambda_ges:', lambda_ges)

L = np.array([71,120,192])
params, covariance_matrix = curve_fit(lambda_fkt,L,noms(lambda_ges)*100)
errors = np.sqrt(np.diag(covariance_matrix))
print('Die Steigung der longitudinalen Mode:')
print('a =', params[0], '±', errors[0])
a = params[0]

plt.cla()
plt.clf()
x = np.linspace(60,200)
plt.plot(L, noms(lambda_ges)*100,'rx',label='Messwerte')
#plt.errorbar(x, noms(lambda_ges)*100, yerr=stds(lambda_ges)*100, label='Messwerte',fmt='rx', capthick=0.5, linewidth='0.5',ecolor='g',capsize=1,markersize=1.5) 
plt.plot(x, lambda_fkt(x,2), 'k-', label='Theorie')
plt.plot(x, lambda_fkt(x,a), 'b-', label='Ausgleichsgerade')
plt.xlabel(r'$L/\si{\centi\meter}$')
plt.ylabel(r'$\Delta \lambda/\si{\centi\meter}$')
plt.xlim(60,200)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.04, w_pad=1.08)
plt.savefig('build/'+'longitudinal')


#transversale Moden
T00x, T00I  = np.genfromtxt('scripts/T00mode.txt',unpack=True) #mm,nA
T00x = 15-T00x
T00x = T00x[::-1]
T00I = T00I[::-1]
T01x, T01I  = np.genfromtxt('scripts/T01mode.txt',unpack=True) #mm.nA
#T00x = T00x /1000 #m
#T01x = T01x /1000 #m

makeTable([T00x,T00I], r'{$\Delta x/ \si{\milli\meter}$} & {$ I / \si{\nano\ampere}$}','tabT00' , ['S[table-format=2.0]' , 'S[table-format=4.0]'] ,  ["%2.0f", "%4.0f"])
makeTable([T01x[:15],T01I[:15]], r'{$ \Delta x / \si{\milli\meter}$} & {$ I/ \si{\nano\ampere}$}','tabT011' , ['S[table-format=1.1]' , 'S[table-format=3.0]'] ,  ["%1.1f", "%3.0f"])
makeTable([T01x[15:],T01I[15:]], r'{$ \Delta x/ \si{\milli\meter}$} & {$ I/ \si{\nano\ampere}$}','tabT012' , ['S[table-format=2.1]' , 'S[table-format=3.0]'] ,  ["%2.1f", "%3.0f"])

#T00 mode fit
def T00(x,a,b,c):
	return a*np.exp(-2*((x-c)**2)/(b**2))
	
params, covariance_matrix = curve_fit(T00,T00x,T00I)
errors = np.sqrt(np.diag(covariance_matrix))
print('Die Parameter der T00 Mode:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('c =', params[2], '±', errors[2])

#plot
x = np.linspace(-0.5, 14.5,10000)
plt.cla()
plt.clf()
plt.plot(x, T00(x,params[0],params[1],params[2]), '-', label='Ausgleichskurve')
plt.plot(T00x, T00I, 'rx', label='Messdaten')
plt.xlim(-0.5, 14.5)
plt.xlabel(r'$\Delta x/\si{\milli\meter}$')
plt.ylabel(r'$I / \si{\nano\ampere}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/'+'T00')

#T01 mode fit
def T01(x,a,b,c):
	return ((x-c)**2)*a*np.exp(-2*((x-c)**2)/(b**2))

params, covariance_matrix = curve_fit(T01,T01x,T01I,p0 = [30,5,5])
errors = np.sqrt(np.diag(covariance_matrix))
print('Die Parameter der T01 mode:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('c =', params[2], '±', errors[2])

#plot
x = np.linspace(-0.5, 15.5,10000)
plt.cla()
plt.clf()
plt.plot(x, T01(x,params[0],params[1],params[2]), '-', label='Ausgleichskurve')
plt.plot(T01x, T01I, 'rx', label='Messdaten')
plt.xlim(-0.5, 15.5)
plt.xlabel(r'$\Delta x/\si{\milli\meter}$')
plt.ylabel(r'$I / \si{\nano\ampere}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/'+'T01')


#Polarisation
Winkelpol, Intenspol  = np.genfromtxt('scripts/polarisation.txt',unpack=True) #grad,mikroA
makeTable([Winkelpol,Winkelpol*2*np.pi/360,Intenspol], r'{$\varphi / \si{\degree} $} & {$\varphi / \text{rad} $} & {$ I / \si{\micro\ampere}$}','tabpolarisation' , ['S[table-format=2.0]','S[table-format=1.2]' , 'S[table-format=2.2]'] ,  ["%2.0f", "%1.2f", "%2.2f"])
Winkelpol = Winkelpol *2 *(np.pi)/360 # rad


def polar(x,a,b,c):
	return a*(np.cos(b*x+c)**2)

params, covariance_matrix = curve_fit(polar,Winkelpol,Intenspol,p0 = [10,1,1])
errors = np.sqrt(np.diag(covariance_matrix))
print('Die Parameter der Polarisationsmessung:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('c =', params[2], '±', errors[2])

x = np.linspace(-0.25, 6.5,10000)
plt.cla()
plt.clf()
plt.plot(x, polar(x,params[0],params[1],params[2]), '-', label='Ausgleichskurve')
plt.plot(Winkelpol, Intenspol, 'rx', label='Messdaten')
plt.xticks([0, np.pi / 4, np.pi/2, 3 * np.pi / 4, np.pi, np.pi*5/4, np.pi*3/2, np.pi*7/4, np.pi*2],
           [r"$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$", r"$\frac{3}{4}\pi$", r"$\pi$", r"$\frac{5}{4}\pi$", r"$\frac{3}{2}\pi$", r"$\frac{7}{4}\pi$", r"$2\pi$"])
plt.xlim(-0.25, 6.5)
plt.xlabel(r'$\varphi$')
plt.ylabel(r'$I/\si{\micro\ampere}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/'+'Polarisation')


#Wellenlängenbestimmung
n, pos = np.genfromtxt('scripts/wellenlaenge.txt', unpack=True) #Umdrehungen
pos = 0.5*pos #mm
pos = pos/1000 #m

g=0.0125/1000
b=55/1000 #Abstand Diode-Gitter

def wellenlaenge(x,n):
	return g*np.sin(np.arctan(np.abs(x)/b))/n

lambdas = wellenlaenge(pos, n)
print('lambdas',lambdas)

makeTable([n, pos*1000, lambdas*10**9], r'{$n$} & {$x_n/ \si{\milli\meter}$} & {$\lambda/ \si{\nano\meter}$}','tabwelle' , ['S[table-format=1.0]', 'S[table-format=2.1]', 'S[table-format=3.2]'] ,  ["%1.0f", "%2.1f", "%3.2f"])

#mittelwert
lambda_m = unp.uarray(*avg_and_sem(lambdas))
print('Mittelwert',lambda_m)
