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


#magnetfeld
B_max = 213*10**(-3) #T
#B_max = B_max*1.75
print('max B field [mT]:', B_max*10**3)
z,B = np.genfromtxt('scripts/magnetfeld.txt',unpack=True) # mm, mT
#B = B*1.75

plt.cla()
plt.clf()
plt.plot(z,B,'rx')
plt.xlabel(r'$z/\si{\milli\metre}$')
plt.ylabel(r'$B/\si{\milli\tesla}$')
#plt.xlim(-30,30)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.legend(loc='best')
plt.savefig('build/magnetfeld.pdf')

makeTable([z[:10], B[:10]], r'{$z/\si{\milli\metre}$} & {$B/\si{\milli\tesla}$}', 'magnetfeld', ['S[table-format=2.0]', 'S[table-format=3.0]'], ["%2.0f", "%3.0f"])
makeTable([z[9:], B[9:]], r'{$z/\si{\milli\metre}$} & {$B/\si{\milli\tesla}$}', 'magnetfeld2', ['S[table-format=2.0]', 'S[table-format=3.0]'], ["%2.0f", "%3.0f"])

#reinprobe
print('reinprobe')
l,t1,t2 = np.genfromtxt('scripts/reinprobe.txt',unpack=True) # micrometer, degree
tr = (t1-t2)/2
tr = 2*np.pi*tr/360 # rad
tr = tr/5.11*10**3 #rad/m

def hyperbel(l,a):
	return a/l**2

params, covariance_matrix = curve_fit(hyperbel,l,tr)
errors = np.sqrt(np.diag(covariance_matrix))

x = np.linspace(0.9,2.7,1000)
plt.cla()
plt.clf()
plt.plot(l**2,tr,'rx',label=r'Messwerte')
plt.plot(x**2,hyperbel(x,*params),'b-',label=r'Ausgleichskurve')
plt.xlabel(r'$\lambda^2/\si{\micro\metre\squared}$')
plt.ylabel(r'$\Delta\theta/\si{\text{rad}\per\metre}$')
plt.xlim(0.9,7.2)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('build/reinprobe.pdf')

print('a=',unp.uarray(params[0],errors[0]))
#print('b=',unp.uarray(params[1],errors[1]))
#print('c=',unp.uarray(params[2],errors[2]))
makeTable([l, t1, t2, tr], r'{$\lambda/\si{\micro\metre}$} & {$\theta_1/\si{\degree}$} & {$\theta_2/\si{\degree}$} & {$\Delta\theta_.{norm}/\si{.{rad}\per\metre}$}', 'undotiert', ['S[table-format=1.2]', 'S[table-format=3.0]', 'S[table-format=3.0]', 'S[table-format=2.0]'], ["%1.2f", "%3.0f", "%3.0f", "%2.0f"])


#dotiert
print('dotiert')
l,t1,t2 = np.genfromtxt('scripts/dotiert1.txt',unpack=True) # micrometer, degree
td1 = (t1-t2)/2
td1 = 2*np.pi*td1/360 # rad
td1 = td1/1.36*10**3 #rad/m

plt.cla()
plt.clf()
plt.plot(l**2,td1,'rx',label=r'Messwerte')
plt.xlabel(r'$\lambda^2/\si{\micro\metre\squared}$')
plt.ylabel(r'$\Delta\theta/\si{\text{rad}\per\metre}$')
#plt.xlim(0.9,7.2)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.legend(loc='best')
plt.savefig('build/dotiert1.pdf')

makeTable([l, t1, t2, td1], r'{$\lambda/\si{\micro\metre}$} & {$\theta_1/\si{\degree}$} & {$\theta_2/\si{\degree}$} & {$\Delta\theta_.{norm}/\si{.{rad}\per\metre}$}', 'dotiert1', ['S[table-format=1.2]', 'S[table-format=3.0]', 'S[table-format=3.0]', 'S[table-format=2.0]'], ["%1.2f", "%3.0f", "%3.0f", "%2.0f"])

l,t1,t2 = np.genfromtxt('scripts/dotiert2.txt',unpack=True) # micrometer, degree
td2 = (t1-t2)/2
td2 = 2*np.pi*td2/360 # rad
td2 = td2/1.296*10**3 #rad/m

plt.cla()
plt.clf()
plt.plot(l**2,td2,'rx',label=r'Messwerte')
plt.xlabel(r'$\lambda^2/\si{\micro\metre\squared}$')
plt.ylabel(r'$\Delta\theta/\si{\text{rad}\per\metre}$')
#plt.xlim(0.9,7.2)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.legend(loc='best')
plt.savefig('build/dotiert2.pdf')

makeTable([l, t1, t2, td2], r'{$\lambda/\si{\micro\metre}$} & {$\theta_1/\si{\degree}$} & {$\theta_2/\si{\degree}$} & {$\Delta\theta_.{norm}/\si{.{rad}\per\metre}$}', 'dotiert2', ['S[table-format=1.2]', 'S[table-format=3.0]', 'S[table-format=3.0]', 'S[table-format=2.0]'], ["%1.2f", "%3.0f", "%3.0f", "%2.0f"])


#dif
def squared(l,a):
	return a*l**2

td1=td1-tr
params1, covariance_matrix = curve_fit(squared,l[1:],td1[1:])
errors1 = np.sqrt(np.diag(covariance_matrix))
td12=td2-tr
params2, covariance_matrix = curve_fit(squared,l[1:],td2[1:])
errors2 = np.sqrt(np.diag(covariance_matrix))

x = np.linspace(0.9,2.7,1000)
plt.cla()
plt.clf()
plt.figure(figsize=(6,4))
plt.plot(l**2,td1,'rx',label=r'Messwerte 1')
plt.plot(l[0]**2,td1[0],'gx')
plt.plot(x**2,squared(x,*params1),'b-',label=r'Ausgleichsgerade 1')
plt.plot(l**2,td2,'mx',label=r'Messwerte 2')
plt.plot(x**2,squared(x,*params2),'c-',label=r'Ausgleichsgerade 2')
plt.plot(l[0]**2,td2[0],'gx',label=r'Ignorierte Messwerte')
plt.xlabel(r'$\lambda^2/\si{\micro\metre\squared}$')
plt.ylabel(r'$\Delta\theta/\si{\text{rad}\per\metre}$')
plt.xlim(0.9,7.2)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('build/dotiert_dif.pdf')

a1=unp.uarray(params1[0],errors1[0])
a2=unp.uarray(params2[0],errors2[0])
print('a1 =', a1)
print('a2 =', a2)


#massen
def masse(a,N,B,n):
	return unp.sqrt(const.e**3*N*B/(a*8*np.pi**2*n*const.c**3*const.epsilon_0))

a1 = a1*10**12
a2 = a2*10**12
N1 = 1.2*10**24
N2 = 2.8*10**24
n = 3.4

print('m1 =', masse(a1,N1,B_max,n))
print('m2 =', masse(a2,N2,B_max,n))
print('m1/me =', masse(a1,N1,B_max,n)/const.m_e)
print('m2/me =', masse(a2,N2,B_max,n)/const.m_e)










