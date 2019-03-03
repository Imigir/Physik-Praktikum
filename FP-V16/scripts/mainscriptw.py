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
import uncertainties
import scipy.constants as const
from errorfunkt2tex import error_to_tex
from errorfunkt2tex import scipy_to_unp
from matplotlib.legend_handler import (HandlerLineCollection,HandlerTuple)
import random

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
# unp.uarray(*weighted_avg_and_sem(unp.nominal_values(bneuDiff), 1/unp.std_devs(bneuDiff))) achtung sum(gewichte muss gleich anzahl der Messungen sein)

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
# params =  uncertainties.correlated_values(params, covar)
# makeNewTable([convert((r'$c_\text{1}$',r'$c_\text{2}$',r'$T_{\text{A}1}$',r'$T_{\text{A}2}$',r'$\alpha$',r'$D_1$',r'$D_2$',r'$A_1$',r'$A_2$',r'$A_3$',r'$A_4$'),strFormat),convert(np.array([paramsGes2[0],paramsGes1[0],deltat2*10**6,deltat1*10**6,-paramsDaempfung[0]*2,4.48*10**-6 *paramsGes1[0]/2*10**3, 7.26*10**-6 *paramsGes1[0]/2*10**3, (VierteMessung-2*deltat2*10**6)[0]*10**-6 *1410 /2*10**3, unp.uarray((VierteMessung[1]-VierteMessung[0])*10**-6 *1410 /2*10**3, 0), unp.uarray((VierteMessung[2]-VierteMessung[1])*10**-6 *2500 /2*10**3, 0),unp.uarray((VierteMessung[3]-VierteMessung[2])*10**-6 *1410 /2*10**3, 0)]),unpFormat,[[r'\meter\per\second',"",True],[r'\meter\per\second',"",True],[r'\micro\second',"",True],[r'\micro\second',"",True],[r'\per\meter',"",True],[r'\milli\meter',"",True],[r'\milli\meter',"",True],[r'\milli\meter',"",True],[r'\milli\meter',r'1.3f',True],[r'\milli\meter',r'1.3f',True],[r'\milli\meter',r'2.2f',True]]),convert(np.array([2730,2730]),floatFormat,[r'\meter\per\second','1.0f',True])+convert((r'-',r'-'),strFormat)+convert(unp.uarray([57,6.05,9.9],[2.5,0,0]),unpFormat,[[r'\per\meter',"",True],[r'\milli\meter',r'1.2f',True],[r'\milli\meter',r'1.2f',True]])+convert((r'-',r'-',r'-',r'-'),strFormat),convert(np.array([(2730-paramsGes2[0])/2730*100,(2730-paramsGes1[0])/2730*100]),unpFormat,[r'\percent','',True])+convert((r'-',r'-'),strFormat)+convert(np.array([(-paramsDaempfung[0]*2-unp.uarray(57,2.5))/unp.uarray(57,2.5)*100,(4.48*10**-6 *paramsGes1[0]/2*10**3-6.05)/6.05*100, (-7.26*10**-6 *paramsGes1[0]/2*10**3+9.90)/9.90*100]),unpFormat,[r'\percent','',True])+convert((r'-',r'-',r'-',r'-'),strFormat)],r'{Wert}&{gemessen}&{Literaturwert\cite{cAcryl},\cite{alphaAcryl}}&{Abweichung}','Ergebnisse', ['c ','c',r'c','c'])
# makeNewTable([convert(peakPos,unpFormat,[r'','1.2f',True]),time],r'\multicolumn{1}{c}{Kanal} & {T/\si{\micro\second}}','tab1', [r'S', r'S'])

#A, B, C = symbols('A B C')
#f = A**3 *B*cos(C)
#f2 = scipy_to_unp(f, [A, B, C])
#AW, BW = unp.uarray([1,2],[0.1,0.2])
#CW = 3
#print(f2(AW, BW, CW))
#print(error_to_tex(f,'f',[AW, BW, CW], [A, B, C],[A, B]))

def Line(x, a, b):
	return a*x+b

def Plot(x=[], y=[], limx=None, limy=None, xname='', yname='', name='', markername='Wertepaare', marker='rx', linear=True, linecolor='b-', linename='Ausgleichsgerade', xscale=1, yscale=1, save=True, Plot=True):
	uParams = None
	if(Plot):
		dx = abs(x[-1]-x[0])
		if(limx==None):
			xplot = np.linspace((x[0]-0.05*dx)*xscale,(x[-1]+0.05*dx)*xscale,1000)
		else:
			xplot = np.linspace(limx[0]*xscale,limx[1]*xscale,1000)
		if(save):
			plt.cla()
			plt.clf()
		plt.errorbar(noms(x)*xscale, noms(y)*yscale, xerr=stds(x)*xscale, yerr=stds(y)*yscale, fmt=marker, markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=markername)
	if(linear == True):
		params, covar = curve_fit(Line, noms(x), noms(y))
		uParams=uncertainties.correlated_values(params, covar)
		if(Plot):
			plt.plot(xplot*xscale, Line(xplot, *params)*yscale, linecolor, label=linename)
	if(Plot):
		if(limx==None):
			plt.xlim((x[0]-0.05*dx)*xscale,(x[-1]+0.05*dx)*xscale)
		else:
			plt.xlim(limx[0]*xscale,limx[1]*xscale)
		if(limy != None):
			plt.ylim(limy[0]*yscale,limy[1]*yscale)
		plt.xlabel(xname)
		plt.ylabel(yname)
		plt.legend(loc='best')
		plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
		if(save):
			plt.savefig('build/'+name+'.pdf')
	if(linear):
		return(uParams)

#Aktivität

A_0 = unp.uarray(330,1)*10**3 #becquerel
tau = unp.uarray(432.6,0.6)*365*24*3600 #s
t = unp.uarray(7.665,0.013)*10**8 #s

A = A_0*unp.exp(-np.log(2)*t/tau)

print('A =', A)

#Energieverlustmessung
print('Energieverlustmessung')

p_ohne,U1_ohne,U2_ohne,U3_ohne = np.genfromtxt('scripts/dataOhne.txt',unpack=True) #p in mbar, U in V
U_ohne_m = []
U_ohne_s = []
for i in range(len(U1_ohne)):
	U_ohne_m = U_ohne_m + [avg_and_sem([U1_ohne[i],U2_ohne[i],U3_ohne[i]])[0]]
	U_ohne_s = U_ohne_s + [avg_and_sem([U1_ohne[i],U2_ohne[i],U3_ohne[i]])[1]]
U_ohne = unp.uarray(U_ohne_m,U_ohne_s)
makeTable([p_ohne,U1_ohne,U2_ohne,U3_ohne,noms(U_ohne),stds(U_ohne)], r'{'+r'$p_\text{ohne}/(\si{\milli\bar})$'+r'} & {'+r'$U_\text{high,ohne}/\si{\volt}$'+r'} & {'+r'$U_\text{low,ohne}/\si{\volt}$'+r'} & {'+r'$U_\text{mid,ohne}/\si{\volt}$'+r'} & \multicolumn{2}{c}{'+r'$\bar{U}_\text{ohne}/(\si{\volt})$'+r'}','tabDataOhne',['S[table-format=3.2]','S[table-format=1.2]','S[table-format=1.2]','S[table-format=1.2]','S[table-format=1.2]','@{${}\pm{}$}S[table-format=1.2]'],["%3.2f","%1.2f","%1.2f","%1.2f","%1.2f","%1.2f"])

#params_ohne, covar = curve_fit(Line, p_ohne, noms(U_ohne))
#uParams=uncertainties.correlated_values(params, covar)

p_mit,U1_mit,U2_mit,U3_mit = np.genfromtxt('scripts/dataMit.txt',unpack=True) #p in mbar, U in V
U_mit_m = []
U_mit_s = []
for i in range(len(U1_mit)):
	U_mit_m = U_mit_m + [avg_and_sem([U1_mit[i],U2_mit[i],U3_mit[i]])[0]]
	U_mit_s = U_mit_s + [avg_and_sem([U1_mit[i],U2_mit[i],U3_mit[i]])[1]]
U_mit = unp.uarray(U_mit_m,U_mit_s)
makeTable([p_mit,U1_mit,U2_mit,U3_mit,noms(U_mit),stds(U_mit)], r'{'+r'$p_\text{mit}/(\si{\milli\bar})$'+r'} & {'+r'$U_\text{high,mit}/\si{\volt}$'+r'} & {'+r'$U_\text{low,mit}/\si{\volt}$'+r'} & {'+r'$U_\text{mid,mit}/\si{\volt}$'+r'} & \multicolumn{2}{c}{'+r'$\bar{U}_\text{mit}/(\si{\volt})$'+r'}','tabDataMit',['S[table-format=3.2]','S[table-format=1.2]','S[table-format=1.2]','S[table-format=1.2]','S[table-format=1.2]','@{${}\pm{}$}S[table-format=1.2]'],["%3.2f","%1.2f","%1.2f","%1.2f","%1.2f","%1.2f"])

plot = True
if(plot):
	plt.cla()
	plt.clf()
name = 'Energieverlust'
uParams_mit = Plot(x=p_mit,y=U_mit,xname=r'$p/\si{\milli\bar}$',yname=r'$U/\si{\volt}$',markername='Wertepaare mit Folie',linename='Ausgleichsgerade mit Folie',save=False,Plot=plot)
uParams_ohne = Plot(x=p_ohne,y=U_ohne,xname=r'$p/\si{\milli\bar}$',yname=r'$U/\si{\volt}$',markername='Wertepaare ohne Folie',linename='Ausgleichsgerade ohne Folie',marker='mx',linecolor='c-',save=False,Plot=plot)
if(plot):
	plt.savefig('build/'+name+'.pdf')

print('steigung mit:', uParams_mit[0], 'V/mbar')
print('achsenabschnitt mit:', uParams_mit[1], 'mbar')
print('steigung ohne:', uParams_ohne[0], 'V/mbar')
print('achsenabschnitt ohne:', uParams_ohne[1], 'mbar')

E_a = 5.486 *10**6 * const.e
m_a = const.value(u"alpha particle mass")
m_e = const.value(u"electron mass")
z = 2
Z = 79
I = 10*Z * const.e
print('I:', I)
n = 19.32/197*const.N_A*1000*1000
print('n:', n)
print(const.N_A)
kappa = E_a/uParams_ohne[1]
DE = (uParams_ohne[1]-uParams_mit[1])*kappa
print('kappa:', kappa/const.e)
print('DE:', DE)
print('DE:', DE/const.e)
v2 = 2*E_a/m_a
print('v2', np.sqrt(v2))
d = DE/np.log(2*m_e*v2/I)*m_e*v2*4*np.pi*const.epsilon_0**2/(const.e**4*z**2*Z*n)
print('d:', d)

sigma_A = np.sqrt(4539)/300
#print(sigma_A)


#Differentieller WQ
dO = np.pi/4/(10.1)**2
A2 = unp.uarray(15.13,sigma_A)
A_exp = A2*np.pi*101**2
print('A_exp= ',A_exp)
print('A2=', A2)
print(dO)

theta,anzahl,t = np.genfromtxt('scripts/dataDeg2.txt',unpack=True)
theta=theta*2*np.pi/360
anzahl = unp.uarray(anzahl,np.sqrt(anzahl))
dsdO=(anzahl/t)/(A2*n*2*10**(-6)*dO)
dsdO2=1/(4*np.pi*const.epsilon_0)**2*((Z*z*const.e**2)/(4*E_a))**2*1/(np.sin(theta*0.5))**4
#print(dsdO)
#print(dsdO2)
makeTable([theta*360/(2*np.pi),noms(anzahl),stds(anzahl),t,noms(dsdO)*10**24,stds(dsdO)*10**24,dsdO2*10**24], r'{'+r'$\theta/\si{\degree}$'+r'} & \multicolumn{2}{c}{'+r'$N$'+r'} & {'+r'$t/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$\left(\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}\right)_\text{exp}/\si{\barn}$'+r'} & {'+r'$\left(\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}\right)_\text{theo}/\si{\barn}$'+r'}','tabDataDeg',['S[table-format=2.1]','S[table-format=4.0]','@{${}\pm{}$}S[table-format=2.0]','S[table-format=3.0]','S[table-format=5.2]','@{${}\pm{}$}S[table-format=2.2]','S[table-format=3.1]'],["%2.1f","%4.0f","%2.0f","%3.0f","%5.2f","%2.2f","%3.1f"])



#plot = False
if(plot):
	plt.cla()
	plt.clf()
	name = 'Rutherford'
	plt.plot(theta, dsdO2*10**24, 'mx', label='Wertepaare Theorie')
	Plot(x=theta,y=dsdO*10**24,xname=r'$\theta$',yname=r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/\si{\barn}$',markername='Wertepaare Experiment',linear=False,save=False,Plot=plot)
	plt.savefig('build/'+name+'.pdf')


#plot = False
if(plot):
	plt.cla()
	plt.clf()
	name = 'Rutherford2'
	plt.plot(theta[1:], dsdO2[1:]*10**24, 'mx', label='Wertepaare Theorie')
	Plot(x=theta[1:],y=dsdO[1:]*10**24,xname=r'$\theta$',yname=r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/\si{\barn}$',markername='Wertepaare Experiment',linear=False,save=False,Plot=plot)
	plt.savefig('build/'+name+'.pdf')

#Z-Abhängigkeit
x=np.array([79,13,83])
dx=np.array([2,3,1])
n_array=np.array([5.9*10**28,6.2*10**28,2.9*10**28])
I=unp.uarray([2.85,0.68,0.35],[0.03,0.06,0.05])
y=I/(n_array*A2*dx*10**(-6)*dO)
y2_30=1/(4*np.pi*const.epsilon_0)**2*((x*z*const.e**2)/(4*E_a))**2*1/(np.sin(1.5*2*np.pi/360))**4
y2_35=1/(4*np.pi*const.epsilon_0)**2*((x*z*const.e**2)/(4*E_a))**2*1/(np.sin(1.75*2*np.pi/360))**4
y2_40=1/(4*np.pi*const.epsilon_0)**2*((x*z*const.e**2)/(4*E_a))**2*1/(np.sin(2*2*np.pi/360))**4

#print('exp:', y)
#print('theo:', y2)

makeTable([x,noms(y)*10**24,stds(y)*10**24,y2_30*10**24,y2_35*10**24,y2_40*10**24],r'{'+r'$Z$'+r'} & \multicolumn{2}{c}{'+r'$\left(\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}\right)_\text{exp}/\si{\barn}$'+r'} & {'+r'$\left(\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}\right)_\text{theo,3}/\si{\barn}$'+r'} & {'+r'$\left(\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}\right)_\text{theo,3.5}/\si{\barn}$'+r'} & {'+r'$\left(\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}\right)_\text{theo,4}/\si{\barn}$'+r'}','tabZAbh',['S[table-format=2.0]','S[table-format=3.0]','@{${}\pm{}$}S[table-format=2.0]','S[table-format=3.2]','S[table-format=3.2]','S[table-format=3.2]'],["%2.0f","%3.0f","%2.0f","%3.2f","%3.2f","%3.2f"])
makeTable([x,dx,n_array/10**28,noms(I),stds(I)],r'{'+r'$Z$'+r'} & {'+r'$d/\si{\micro\metre}$'+r'} & {'+r'$n/10^{28}\si{\metre^{-3}}$'+r'} & \multicolumn{2}{c}{'+r'$I_\theta/\si{\second^{-1}}$'+r'}','tabZWerte',['S[table-format=2.0]','S[table-format=1.0]','S[table-format=1.1]','S[table-format=1.2]','@{${}\pm{}$}S[table-format=1.2]'],["%2.0f","%1.0f","%1.1f","%1.2f","%1.2f"])


plt.cla()
plt.clf()
plt.errorbar(noms(x), noms(y)*10**24, xerr=stds(x), yerr=stds(y)*10**24, fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x,y2_30*10**24,'bx',label=r'Theorie $\theta=3\si{\degree}$')
plt.plot(x,y2_35*10**24,'kx',label=r'Theorie $\theta=3,5\si{\degree}$')
plt.plot(x,y2_40*10**24,'yx',label=r'Theorie $\theta=4\si{\degree}$')
plt.xlabel(r'$Z$')
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/\si{\barn}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('build/zAbh.pdf')
