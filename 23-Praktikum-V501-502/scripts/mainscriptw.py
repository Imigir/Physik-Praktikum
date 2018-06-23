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


#Magnetfeld
R=0.282
L=17.5/100
N=20
my=const.mu_0
D,I_350,I_250 = np.genfromtxt('scripts/data7.txt', unpack=True)
D=D*6.25/1000
r=D/(L**2+D**2)
#print(r)
B_350=my*8/np.sqrt(125)*N*I_350/R
B_250=my*8/np.sqrt(125)*N*I_250/R

def rad(B,a,c):
    return a*B+c

params1,covar1,sigma_y=linregress(B_350,r)
params2,covar2,sigma_y=linregress(B_250,r)

B_plot=np.linspace(0,22/10**5,1000)
plt.cla()
plt.clf()
plt.plot(B_plot*10**5,rad(B_plot,*params2),'b-',label=r'Ausgleichsgerade:$U_\text{B} = \SI{250}{\volt}$')
plt.plot(B_plot*10**5,rad(B_plot,*params1),'y-',label=r'Ausgleichsgerade:$U_\text{B} = \SI{350}{\volt}$')
plt.plot(B_250*10**5, r, 'rx', label=r'Messwerte:$U_\text{B} = \SI{250}{\volt}$')
plt.plot(B_350*10**5, r, 'gx', label=r'Messwerte:$U_\text{B} = \SI{350}{\volt}$')
plt.ylabel(r'$\frac{D}{D^2+L^2}/\si{\metre}$')
plt.xlabel(r'$B/10^{-5}\si{\tesla}$')
plt.xlim(0,22)
plt.legend(loc='best')
plt.savefig('content/images/GraphMag1.pdf')

a_250=unp.uarray(params2[0],covar2[0])
b_250=unp.uarray(params2[1],covar2[1])

a_350=unp.uarray(params1[0],covar1[0])
b_350=unp.uarray(params1[1],covar1[1])

print('Steigung 250V: ', a_250)
print('Achsenabschnitt 250V: ', b_250)
print('Steigung 350V: ',a_350)
print('Achsenabschnitt 350V: ',b_350)

e_m_250=8*a_250**2*250
e_m_350=8*a_350**2*350

print('e0/m0_250: ',e_m_250)
print('e0/m0_350: ',e_m_350)

B_erde=my*8/np.sqrt(125)*N*0.235/R/np.cos(7/36*2*np.pi)

print('B_erde_150: ', B_erde)
makeTable([D*1000,r,I_250,B_250,I_350,B_350], r'{'+r'$D/10^{-3}\si{\metre}$'+r'} & {'+r'$\frac{D}{D^2+L^2}/\si{\metre}$'+r'} & {'+r'$I_.{250}/\si{\ampere}$'+r'} & {'+r'$B_.{250}/10^{-5}\si{\tesla}$'+r'} & {'+r'$I_.{350}/\si{\ampere}$'+r'} & {'+r'$B_.{350}/10^{-5}\si{\tesla}$'+r'}', 'tabMag',['S[table-format=2.2]','S[table-format=1.2]','S[table-format=1.2]','S[table-format=2.2]','S[table-format=1.2]','S[table-format=2.2]'],["%2.2f","%1.2f","%1.2f","%2.2f","%1.2f","%2.2f"])


# E-Feld

L2=(14.3)/100
p=0.019
d=0.00665
k=p*L2/2/d
D_E= np.genfromtxt('scripts/data1.txt',unpack=True)
D_E=-D_E*6.25/1000
U_1=np.genfromtxt('scripts/data2.txt', unpack=True)
U_2=np.genfromtxt('scripts/data3.txt', unpack=True)
U_3=np.genfromtxt('scripts/data4.txt', unpack=True)
U_4=np.genfromtxt('scripts/data5.txt', unpack=True)
U_5=np.genfromtxt('scripts/data6.txt', unpack=True)

makeTable([D_E*1000,U_1,U_2,U_3,U_4,U_5],r'{'+r'$D/\si{\metre}$'+r'} & {'+r'$U_.{d,310}/\si{\volt}$'+r'} & {'+r'$U_.{d,280}/\si{\volt}$'+r'} & {'+r'$U_.{d,260}/\si{\volt}$'+r'} & {'+r'$U_.{d,240}/\si{\volt}$'+r'} & {'+r'$U_.{d,220}/\si{\volt}$'+r'}','tabElek',['S[table-format=2.2]','S[table-format=2.1]','S[table-format=2.1]','S[table-format=2.1]','S[table-format=2.1]','S[table-format=2.1]'],["%2.2f","%2.1f","%2.1f","%2.1f","%2.1f","%2.1f"])

def Dfunc(U,c,d):
    return c*U+d

para1,cova1,sigma_y=linregress(U_1,D_E)
para2,cova2,sigma_y=linregress(U_2,D_E)
para3,cova3,sigma_y=linregress(U_3,D_E)
para4,cova4,sigma_y=linregress(U_4,D_E)
para5,cova5,sigma_y=linregress(U_5,D_E)

U_plot=np.linspace(-14,35,1000)
plt.cla()
plt.clf()
plt.plot(U_plot,Dfunc(U_plot,*para1),'b-',label=r'$U_\text{B} = \SI{310}{\volt}$')
plt.plot(U_1, D_E, 'bx')
plt.plot(U_plot,Dfunc(U_plot,*para2),'g-',label=r'$U_\text{B} = \SI{280}{\volt}$')
plt.plot(U_2, D_E, 'gx')
plt.plot(U_plot,Dfunc(U_plot,*para3),'y-',label=r'$U_\text{B} = \SI{260}{\volt}$')
plt.plot(U_3, D_E, 'yx')
plt.plot(U_plot,Dfunc(U_plot,*para4),'r-',label=r'$U_\text{B} = \SI{240}{\volt}$')
plt.plot(U_4, D_E, 'rx')
plt.plot(U_plot,Dfunc(U_plot,*para5),'k-',label=r'$U_\text{B} = \SI{220}{\volt}$')
plt.plot(U_5, D_E, 'kx')
plt.ylabel(r'$D/\si{\metre}$')
plt.xlabel(r'$U_d/\si{\volt}$')
plt.legend(loc='best')
plt.savefig('content/images/GraphEle.pdf')



a1=unp.uarray(para1[0],cova1[0])
a2=unp.uarray(para2[0],cova2[0])
a3=unp.uarray(para3[0],cova3[0])
a4=unp.uarray(para4[0],cova4[0])
a5=unp.uarray(para5[0],cova5[0])

b1=unp.uarray(para1[1],cova1[1])
b2=unp.uarray(para2[1],cova2[1])
b3=unp.uarray(para3[1],cova3[1])
b4=unp.uarray(para4[1],cova4[1])
b5=unp.uarray(para5[1],cova5[1])

a_array=np.array([para1[0],para2[0],para3[0],para4[0],para5[0]])
b_array=np.array([para1[1],para2[1],para3[1],para4[1],para5[1]])
af_array=np.array([cova1[0],cova2[0],cova3[0],cova4[0],cova5[0]])
bf_array=np.array([cova1[1],cova2[1],cova3[1],cova4[1],cova5[1]])
U_Barray=np.array([310,280,260,240,220])
print(a_array)
print(af_array)
print(b_array)
print(bf_array)
def afunc(U,e,f):
    return e/U+f
para6,cova6,sigma_y=linregress(1/U_Barray,a_array)
U2_plot=np.linspace(218,312,1000)


plt.cla()
plt.clf()
plt.plot(1/U2_plot,afunc(U2_plot,*para6),'b-',label=r'Ausgleichsgerade')
plt.plot(1/U_Barray, a_array, 'rx', label=r'Messwerte')
plt.ylabel(r'$\alpha/\si{\metre\per\volt}$')
plt.xlabel(r'$U_B^{-1}/\si{\volt}$')
plt.legend(loc='best')
plt.savefig('content/images/GraphEle6.pdf')

a_ges=unp.uarray(para6[0],cova6[0])
b_ges=unp.uarray(para6[1],cova6[1])
Abweichung=(para6[0]/k-1)*100
print('Steigung der a: ', a_ges)
print('k: ',k)
print('Abweichung von der Theorie in %:',Abweichung)
print('y-Achsenabschnitt: ',b_ges)

# Frequenz
Amp=0.00625*1.5
n,f=np.genfromtxt('scripts/data8.txt',unpack=True)
#makeTable([1/n,f,f/n],r'{'+r'$n$'+r'} & {'+r'$f_.{Säge}/\si{\hertz}$'+r'} & {'+r'$f_.{sin}/\si{\hertz}$'+r'}','tabFreq',['S[table-format=1.1]','S[table-format=2.2]', 'S[table-format=2.2]'],["%1.1f","%2.2f","%2.2f"])
f_avg=unp.uarray(avg_and_sem(f/n)[0],avg_and_sem(f/n)[1])
empfind=a_ges/350
S=Amp*350/(a_ges)
S2=Amp/empfind
print('Mittelwert der Sinus-Frequenz: ',f_avg)
print('Scheitelpunkt der Sinusspannung: ',S, 'oder ', S2)
