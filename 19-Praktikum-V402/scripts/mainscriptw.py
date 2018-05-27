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

# Tabellen

phi_r, phi_l, phi = np.genfromtxt('scripts/data2.txt',unpack=True)
o_r, o_l, eta = np.genfromtxt('scripts/data1.txt', unpack=True)
phi_avg= avg_and_sem(phi)
print('Der Mittlere Winkel Phi ist:',phi_avg,'°')
eta = eta/360*2*np.pi
n=np.sin((eta+phi_avg[0]*2*np.pi/360)/2)/np.sin(phi_avg[0]*2*np.pi/360/2)

makeTable([phi_r, phi_l, phi], r'{'+r'$\phi_.r/\si{\degree}$'+r'} & {'+r'$\phi_.l/\si{\degree}$'+r'} & {'+r'$\phi/\si{\degree}$'+r'}', 'tabphi',['S[table-format=3.1]','S[table-format=3.1]','S[table-format=2.2]'],["%3.1f", "%3.1f", "%2.2f"])

#n_err = np.sin(eta/2)*phi_avg[1]*2*np.pi/360/(np.cos(phi_avg[0]*2*np.pi/360)-1)
n_err = -1/2*np.sin(eta/2)*phi_avg[1]*2*np.pi/360/(np.sin(phi_avg[0]*np.pi/360)**2)
makeTable([o_r, o_l, eta/2/np.pi*360, n, np.abs(n_err)], r'{'+r'$\omega_.r/\si{\degree}$'+r'} & {'+r'$\omega_.l/\si{\degree}$'+r'} & {'+r'$\eta/\si{\degree}$'+r'} &  \multicolumn{2}{c}{'+r'$n$'+r'}', 'tabn', ['S[table-format=3.1]','S[table-format=3.1]','S[table-format=2.1]','S[table-format=1.2]','@{${}\pm{}$}S[table-format=1.5]'],["%3.1f", "%3.1f", "%2.1f", "%1.2f","%1.5f"])


# Graphen
lambd_a = np.genfromtxt('scripts/data3.txt',unpack=True)
lambd_a = lambd_a/10**9
def n1(l,a,b):
    return a+b/(l**2)

lplot = np.linspace(400/10**9,700/10**9,1000)
params, covar = curve_fit(n1, lambd_a, n**2)
plt.cla()
plt.clf()
plt.plot(lplot*10**9,n1(lplot, *params),'b-', label='Ausgleichskurve')
plt.plot(lambd_a*10**9, n**2,'rx', label='Messwerte')
plt.xlabel(r'$\lambda/10^{-9}\si{\metre}$')
plt.ylabel(r'$n^2$')
plt.xlim(400,700)
plt.ylim(1.725**2,1.85**2)
plt.legend(loc='best')
plt.savefig('content/images/Graph11.pdf')


A_0 = unp.uarray(params[0], np.sqrt(covar[0][0]))
A_2 = unp.uarray(params[1], np.sqrt(covar[1][1]))
print('A0 von Gleichung 11:', A_0)
print('A2 von Gleichung 11:', A_2)
s1=0
Wert_A_0=params[0]
Wert_A_2=params[1]
for i in range(len(n)):
    s1=s1+(n[i]**2-Wert_A_0-Wert_A_2/lambd_a[i]**2)

s1=s1**2/8

print('Quadrate der Abweichung von Gleichung 11:',s1)

def n2(l,a,b):
    return a-b*l**2

params2, covar2 = curve_fit(n2, lambd_a, n**2)
plt.cla()
plt.clf()
plt.plot(lplot*10**9,n2(lplot, *params2),'b-', label='Ausgleichskurve')
plt.plot(lambd_a*10**9, n**2,'rx', label='Messwerte')
plt.xlabel(r'$\lambda/10^{-9}\si{\metre}$')
plt.ylabel(r'$n^2$')
plt.xlim(400,700)
plt.ylim(1.725**2,1.85**2)
plt.legend(loc='best')
plt.savefig('content/images/Graph11a.pdf')

A_0a = unp.uarray(params2[0], np.sqrt(covar2[0][0]))
A_2a = unp.uarray(params2[1], np.sqrt(covar2[1][1]))
print('A0 von Gleichung 11a:', A_0a)
print('A2 von Gleichung 11a:', A_2a)
s2=0
Wert_A_0a= params2[0]
Wert_A_2a= params2[1]
for i in range(len(n)):
    s2=s2+(n[i]**2-Wert_A_0a-Wert_A_2a/lambd_a[i]**2)

s2=s2**2/8
print('Abweichungsquadrate der Gleichung 11a:',s2)

def n3(l,a,b,c):
    return a+b/(l**2)+c/(l**4)

params3, covar3 = curve_fit(n3, lambd_a, n**2)
plt.cla()
plt.clf()
plt.plot(lplot*10**9,n3(lplot, *params3),'b-', label='Ausgleichskurve')
plt.plot(lambd_a*10**9, n**2,'rx', label='Messwerte')
plt.xlabel(r'$\lambda/10^{-9}\si{\metre}$')
plt.ylabel(r'$n^2$')
plt.xlim(400,700)
plt.ylim(1.725**2,1.85**2)
plt.legend(loc='best')
plt.savefig('content/images/Graph11.4.pdf')

A_0_4 = unp.uarray(params3[0], np.sqrt(covar3[0][0]))
A_2_4 = unp.uarray(params3[1], np.sqrt(covar3[1][1]))
A_4_4 = unp.uarray(params3[2], np.sqrt(covar3[2][2]))
print('A0 von der optimierten Gleichung 11:', A_0_4)
print('A2 von der optimierten Gleichung 11:', A_2_4)
print('A4 von der optimierten Gleichung 11:', A_4_4)

def n4(l,a,b,c,d):
    return a+b/(l**2)+c/(l**4)-d*l**2

params4, covar4 = curve_fit(n4, lambd_a, n**2)
plt.cla()
plt.clf()
plt.plot(lplot*10**9,n4(lplot, *params4),'b-', label='Ausgleichskurve')
plt.plot(lambd_a*10**9, n**2,'rx', label='Messwerte')
plt.xlabel(r'$\lambda/10^{-9}\si{\metre}$')
plt.ylabel(r'$n^2$')
plt.xlim(400,700)
plt.ylim(1.725**2,1.85**2)
plt.legend(loc='best')
plt.savefig('content/images/Graph11.42.pdf')

A_0_4_2 = unp.uarray(params4[0], np.sqrt(covar4[0][0]))
A_2_4_2 = unp.uarray(params4[1], np.sqrt(covar4[1][1]))
A_4_4_2 = unp.uarray(params4[2], np.sqrt(covar4[2][2]))
A_2__4_2 = unp.uarray(params4[3], np.sqrt(covar4[3][3]))

print('A0 von der sehr optimierten Gleichung 11:', A_0_4_2)
print('A2 von der sehr optimierten Gleichung 11:', A_2_4_2)
print('A4 von der sehr optimierten Gleichung 11:', A_4_4_2)
print('A2` von der sehr optimierten Gleichung 11:', A_2__4_2)


#Abbe Zahl
l_C=656/10**9
l_D=589/10**9
l_F=486/10**9
nC=np.sqrt(params[0]+params[1]/l_C**2)
nD=np.sqrt(params[0]+params[1]/l_D**2)
nF=np.sqrt(params[0]+params[1]/l_F**2)

v=(nD-1)/(nF-nC)

print('Frauenhofer Brechungsindizees:',nC,',',nD,',',nF)
print('Abbesche Zahl: ',v)

#Auflösungsvermögen
b=0.03
A_C=-b*params[1]/(np.sqrt(params[0]+params[1]/l_C**2)*l_C**3)
A_F=-b*params[1]/(np.sqrt(params[0]+params[1]/l_F**2)*l_F**3)

print('Auflösungsvermögen A_C:', A_C)
print('Auflösungsvermögen A_F:', A_F)
