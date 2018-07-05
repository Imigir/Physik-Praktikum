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

#a)

alpha,N_bragg = np.genfromtxt(r'scripts/3.7.18Bragg.txt',unpack=True)

makeTable([alpha,N_bragg], r'{'+r'$\alpha/\si{\degree}$'+r'} & {'+r'$N/\si{1\per\second}$'+r'}', 'tabBragg',['S[table-format=2.1]', 'S[table-format=3.0]'],["%2.1f","%3.0f"])

d=201.4*10**(-12)
l=2*d*np.sin(np.pi/36)
E=const.h*const.c/l
print('Lambda: ', l)
print('Energie: ', E)

'''
plt.cla()
plt.clf()
plt.plot(alpha,N_bragg,'rx',label='Messwerte')
plt.xlabel(r'$\alpha/\si{\degree}$')
plt.ylabel(r'$N/\si{1\per\second}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/bragg.png')

print('bragg done')
'''
#b)

theta_Spek,N_Spek = np.genfromtxt('scripts/3.7.18Spektrum.txt',unpack=True)
makeTable([theta_Spek,N_Spek], r'{'+r'$\theta_.{Spek}/\si{\degree}$'+r'} & {'+r'$N/\si{1\per\second}$'+r'}', 'tabSpektrum',['S[table-format=2.1]', 'S[table-format=4.0]'],["%2.1f","%4.0f"])


'''
plt.cla()
plt.clf()
plt.plot(theta_Spek,N_Spek,'rx',label='Messwerte')
plt.xlabel(r'$\theta_{Spek}/\si{\degree}$')
plt.ylabel(r'$N_{Spek}/\si{1\per\second}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/Spektrum.png')

print('spektrum done')
'''

plt.cla()
plt.clf()
plt.plot(theta_Spek[0:156],N_Spek[0:156],'rx',label='Messwerte')
plt.xlabel(r'$\theta_{Spek}/\si{\degree}$')
plt.ylabel(r'$N_{Spek}/\si{1\per\second}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/konstSpektrum.png')

def f(x):
    return 6540*x-129126
def g(x):
    return -3990*x+81082
def const1(x):
    return 540+0*x
a_plot=np.linspace(19.5,20)
b_plot=np.linspace(19.9,20.5)
a_plot2=np.linspace(21.9,22.3)
b_plot2=np.linspace(22.2,22.6)
def f2(x):
    return 20140*x-442454
def g2(x):
    return -15600*x+352019
def const2(x):
    return 1929+0*x
schnitt_plot=np.linspace(19.7,20.3)
schnitt_plot2=np.linspace(22.0,22.5)
plt.cla()
plt.clf()
plt.plot(schnitt_plot2,const2(schnitt_plot),'k-')
plt.plot(schnitt_plot,const1(schnitt_plot),'k-')
plt.plot(a_plot,f(a_plot),'k-')
plt.plot(b_plot,g(b_plot),'k-')
plt.plot(a_plot2,f2(a_plot2),'k-')
plt.plot(b_plot2,g2(b_plot2),'k-')
plt.plot(22.44167,const2(22.44166),'bx')
plt.plot(22.0646971,const2(22.0646971),'bx')
plt.plot(19.8266055,const1(19.8266055),'bx')
plt.plot(20.1859649,const1(20.1859649),'bx')
plt.plot(theta_Spek[156:221],N_Spek[156:221],'rx',label='Messwerte')
plt.xlabel(r'$\theta_{Spek}/\si{\degree}$')
plt.ylabel(r'$N_{Spek}/\si{1\per\second}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/PeakSpektrum.png')

DEa=const.value("Planck constant")*const.c/(2*d*np.sin((20.1859649-19.8266055)*np.pi/180))
DEb=const.value("Planck constant")*const.c/(2*d*np.sin((22.44167-22.0646971)*np.pi/180))
print('b-: ', 19.8266055)
print('b+: ', 20.1859649)
print('a-: ', 22.0646971)
print('a+: ', 22.44167)
print('Breite b: ',DEa)
print('Breite a: ',DEb)
E_beta=const.value("Planck constant")*const.c/(2*d*np.sin((20)*np.pi/180))/const.e
E_alpha=const.value("Planck constant")*const.c/(2*d*np.sin((22.2)*np.pi/180))/const.e
print('E_K_a: ', E_alpha)
print('E_K_b: ', E_beta)
z=29
E_R=13.7
s_K=z-np.sqrt(E_beta/E_R)
s_L=z-2*np.sqrt((E_R*(z-s_K)**2-E_alpha)/E_R)
s_M=z-3*np.sqrt((z-s_K)**2-E_beta/E_R)

print('s_K: ',s_K)
print('s_L: ',s_L)
print('s_M: ',s_M)

#c)
theta_Br,N_Br = np.genfromtxt('scripts/3.7.18Brom.txt',unpack=True)
theta_Sr,N_Sr = np.genfromtxt('scripts/3.7.18Strontium.txt',unpack=True)
theta_Zn,N_Zn = np.genfromtxt('scripts/3.7.18Zink.txt',unpack=True)
theta_Zr,N_Zr = np.genfromtxt('scripts/3.7.18Zirkonium.txt',unpack=True)
theta_Bi,N_Bi = np.genfromtxt('scripts/3.7.18Wismuth.txt',unpack=True)

makeTable([theta_Br,N_Br], r'{'+r'$\theta_.{Br}/\si{\degree}$'+r'} & {'+r'$N/\si{1\per\second}$'+r'}', 'tabBr',['S[table-format=2.1]', 'S[table-format=3.0]'],["%2.1f","%3.0f"])
makeTable([theta_Sr,N_Sr], r'{'+r'$\theta_.{Sr}/\si{\degree}$'+r'} & {'+r'$N/\si{1\per\second}$'+r'}', 'tabSr',['S[table-format=2.1]', 'S[table-format=3.0]'],["%2.1f","%3.0f"])
makeTable([theta_Zn,N_Zn], r'{'+r'$\theta_.{Zn}/\si{\degree}$'+r'} & {'+r'$N/\si{1\per\second}$'+r'}', 'tabZn',['S[table-format=2.1]', 'S[table-format=3.0]'],["%2.1f","%3.0f"])
makeTable([theta_Zr,N_Zr], r'{'+r'$\theta_.{Zr}/\si{\degree}$'+r'} & {'+r'$N/\si{1\per\second}$'+r'}', 'tabZr',['S[table-format=2.1]', 'S[table-format=3.0]'],["%2.1f","%3.0f"])
makeTable([theta_Bi,N_Bi], r'{'+r'$\theta_.{Bi}/\si{\degree}$'+r'} & {'+r'$N/\si{1\per\second}$'+r'}', 'tabBi',['S[table-format=2.1]', 'S[table-format=3.0]'],["%2.1f","%3.0f"])
'''
plt.cla()
plt.clf()
plt.plot(theta_Br,N_Br,'rx',label='Messwerte')
plt.xlabel(r'$\theta_{Br}/\si{\degree}$')
plt.ylabel(r'$N_{Br}/\si{1\per\second}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/Br.png')

print('br done')

plt.cla()
plt.clf()
plt.plot(theta_Sr,N_Sr,'rx',label='Messwerte')
plt.xlabel(r'$\theta_{Sr}/\si{\degree}$')
plt.ylabel(r'$N_{Sr}/\si{1\per\second}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/Sr.png')

print('sr done')

plt.cla()
plt.clf()
plt.plot(theta_Zn,N_Zn,'rx',label='Messwerte')
plt.xlabel(r'$\theta_{Zn}/\si{\degree}$')
plt.ylabel(r'$N_{Zn}/\si{1\per\second}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/Zn.png')

print('zn done')

plt.cla()
plt.clf()
plt.plot(theta_Zr,N_Zr,'rx',label='Messwerte')
plt.xlabel(r'$\theta_{Zr}/\si{\degree}$')
plt.ylabel(r'$N_{Zr}/\si{1\per\second}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/Zr.png')

print('zr done')

plt.cla()
plt.clf()
plt.plot(theta_Bi,N_Bi,'rx',label='Messwerte')
plt.xlabel(r'$\theta_.{Bi}/\si{\degree}$')
plt.ylabel(r'$N_.{Bi}/\si{1\per\second}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/Bi.png')

print('bi done')
'''
