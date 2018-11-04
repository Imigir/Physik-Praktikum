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


#TL
VTL = unp.uarray(10.197,0.915)
#TL1
pTL1,tTL1_1,tTL1_2,tTL1_3 = np.genfromtxt(r'scripts/dataTL1.txt',unpack=True)
p0TL1 = np.array(2*10**(-4))
pTL1 = np.append(p0TL1,pTL1)
pTL1 = unp.uarray(pTL1,pTL1*0.1)

tTL1 = [0]
tTL1_f = [0]

for i in range (0,len(tTL1_1)):
	th,th_f = avg_and_sem([tTL1_1[i],tTL1_2[i],tTL1_3[i]])
	tTL1.append(th)
	tTL1_f.append(th_f)
	
tTL1=unp.uarray(tTL1,tTL1_f)

paramsLinearTL1, errorsLinearTL1, sigma_y = linregress(noms(tTL1), noms(pTL1))
steigungTL1 = unp.uarray(paramsLinearTL1[0],errorsLinearTL1[0])/10**4

STL1 = VTL/p0TL1*steigungTL1
print('STL1 =', STL1)

#TL2
pTL2,tTL2_1,tTL2_2,tTL2_3 = np.genfromtxt(r'scripts/dataTL2.txt',unpack=True)
p0TL2 = np.array(1.4*10**(-4))
pTL2 = np.append(p0TL2,pTL2)
pTL2 = unp.uarray(pTL2,pTL2*0.1)

tTL2 = [0]
tTL2_f = [0]

for i in range (0,len(tTL2_1)):
	th,th_f = avg_and_sem([tTL2_1[i],tTL2_2[i],tTL2_3[i]])
	tTL2.append(th)
	tTL2_f.append(th_f)
	
tTL2=unp.uarray(tTL2,tTL2_f)

paramsLinearTL2, errorsLinearTL2, sigma_y = linregress(noms(tTL2), noms(pTL2))
steigungTL2 = unp.uarray(paramsLinearTL2[0],errorsLinearTL2[0])/10**4

STL2 = VTL/p0TL2*steigungTL2
print('STL2 =', STL2)

#TL3
pTL3,tTL3_1,tTL3_2,tTL3_3 = np.genfromtxt(r'scripts/dataTL3.txt',unpack=True)
p0TL3 = np.array(1*10**(-4))
pTL3 = np.append(p0TL3,pTL3)
pTL3 = unp.uarray(pTL3,pTL3*0.1)

tTL3 = [0]
tTL3_f = [0]

for i in range (0,len(tTL3_1)):
	th,th_f = avg_and_sem([tTL3_1[i],tTL3_2[i],tTL3_3[i]])
	tTL3.append(th)
	tTL3_f.append(th_f)
	
tTL3=unp.uarray(tTL3,tTL3_f)

paramsLinearTL3, errorsLinearTL3, sigma_y = linregress(noms(tTL3), noms(pTL3))
steigungTL3 = unp.uarray(paramsLinearTL3[0],errorsLinearTL3[0])/10**4

STL3 = VTL/p0TL3*steigungTL3
print('STL3 = ', STL3)

#TL4
pTL4,tTL4_1,tTL4_2,tTL4_3 = np.genfromtxt(r'scripts/dataTL4.txt',unpack=True)
p0TL4 = np.array(0.5*10**(-4))
pTL4 = np.append(p0TL4,pTL4)
pTL4 = unp.uarray(pTL4,pTL4*0.1)

tTL4 = [0]
tTL4_f = [0]

for i in range (0,len(tTL4_1)):
	th,th_f = avg_and_sem([tTL4_1[i],tTL4_2[i],tTL4_3[i]])
	tTL4.append(th)
	tTL4_f.append(th_f)
	
tTL4=unp.uarray(tTL4,tTL4_f)

paramsLinearTL4, errorsLinearTL4, sigma_y = linregress(noms(tTL4), noms(pTL4))
steigungTL4 = unp.uarray(paramsLinearTL4[0],errorsLinearTL4[0])/10**4

STL4 = VTL/p0TL4*steigungTL4
print('STL4 = ', STL4)

"""
#Plot
#TL1
plt.cla()
plt.clf()
x_plot = np.linspace(-2,15)

plt.errorbar(noms(tTL1), noms(pTL1), xerr=stds(tTL1), yerr=stds(pTL1), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot,x_plot*paramsLinearTL1[0]+paramsLinearTL1[1],'b-',label='Ausgleichsgerade')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$p/10^{-4}\si{\milli\bar}$')
plt.xlim(-1,15)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/TL1.png')

print('TL1 done')

#TL2
plt.cla()
plt.clf()
x_plot = np.linspace(-2,25)

plt.errorbar(noms(tTL2), noms(pTL2), xerr=stds(tTL2), yerr=stds(pTL2), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot,x_plot*paramsLinearTL2[0]+paramsLinearTL2[1],'b-',label='Ausgleichsgerade')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$p/10^{-4}\si{\milli\bar}$')
plt.xlim(-1,22)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/TL2.png')

print('TL2 done')

#TL3
plt.cla()
plt.clf()
x_plot = np.linspace(-2,35)

plt.errorbar(noms(tTL3), noms(pTL3), xerr=stds(tTL3), yerr=stds(pTL3), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot,x_plot*paramsLinearTL3[0]+paramsLinearTL3[1],'b-',label='Ausgleichsgerade')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$p/10^{-4}\si{\milli\bar}$')
plt.xlim(-1,31)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/TL3.png')

print('TL3 done')

#TL4
plt.cla()
plt.clf()
x_plot = np.linspace(-2,30)

plt.errorbar(noms(tTL4), noms(pTL4), xerr=stds(tTL4), yerr=stds(pTL4), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot,x_plot*paramsLinearTL4[0]+paramsLinearTL4[1],'b-',label='Ausgleichsgerade')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$p/10^{-4}\si{\milli\bar}$')
plt.xlim(-1,29)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/TL4.png')

print('TL4 done')

#TL 1-4
plt.cla()
plt.clf()
x_plot = np.linspace(-2,35)

#TL1
plt.errorbar(noms(tTL1), noms(pTL1), xerr=stds(tTL1), yerr=stds(pTL1), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=2*10^{-4} \si{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearTL1[0]+paramsLinearTL1[1],'k-',label='Ausgleichsgeraden')
#TL2
plt.errorbar(noms(tTL2), noms(pTL2), xerr=stds(tTL2), yerr=stds(pTL2), fmt='bx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=1.4*10^{-4} \si{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearTL2[0]+paramsLinearTL2[1],'k-')
#TL3
plt.errorbar(noms(tTL3), noms(pTL3), xerr=stds(tTL3), yerr=stds(pTL3), fmt='cx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=1*10^{-4} \si{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearTL3[0]+paramsLinearTL3[1],'k-')
#TL3
plt.errorbar(noms(tTL4), noms(pTL4), xerr=stds(tTL4), yerr=stds(pTL4), fmt='mx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=0.5*10^{-4} \si{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearTL4[0]+paramsLinearTL4[1],'k-')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$p/10^{-4}\si{\milli\bar}$')
plt.xlim(-1,31)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/TL.png')
"""
print('TL done')



#DL
VDL = unp.uarray(11.15,1.032)
#DL1
pDL1,tDL1_1,tDL1_2,tDL1_3,tDL1_4 = np.genfromtxt(r'scripts/dataDL1.txt',unpack=True)
p0DL1 = np.array(1)
pDL1 = np.append(p0DL1,pDL1)
pDL1 = unp.uarray(pDL1,pDL1*0.1)

tDL1 = [0]
tDL1_f = [0]

for i in range (0,len(tDL1_1)):
	th,th_f = avg_and_sem([tDL1_1[i],tDL1_2[i],tDL1_3[i],tDL1_4[i]])
	tDL1.append(th)
	tDL1_f.append(th_f)
	
tDL1=unp.uarray(tDL1,tDL1_f)

paramsLinearDL1, errorsLinearDL1, sigma_y = linregress(noms(tDL1), noms(pDL1))
steigungDL1 = unp.uarray(paramsLinearDL1[0],errorsLinearDL1[0])

SDL1 = VDL/p0DL1*steigungDL1
print('SDL1 = ', SDL1)

#DL2
pDL2,tDL2_1,tDL2_2,tDL2_3 = np.genfromtxt(r'scripts/dataDL2.txt',unpack=True)
p0DL2 = np.array(0.8)
pDL2 = np.append(p0DL2,pDL2)
pDL2 = unp.uarray(pDL2,pDL2*0.1)

tDL2 = [0]
tDL2_f = [0]

for i in range (0,len(tDL2_1)):
	th,th_f = avg_and_sem([tDL2_1[i],tDL2_2[i],tDL2_3[i]])
	tDL2.append(th)
	tDL2_f.append(th_f)
	
tDL2=unp.uarray(tDL2,tDL2_f)

paramsLinearDL2, errorsLinearDL2, sigma_y = linregress(noms(tDL2), noms(pDL2))
steigungDL2 = unp.uarray(paramsLinearDL2[0],errorsLinearDL2[0])

SDL2 = VDL/p0DL2*steigungDL2
print('SDL2 = ', SDL2)

#DL3
pDL3,tDL3_1,tDL3_2,tDL3_3 = np.genfromtxt(r'scripts/dataDL3.txt',unpack=True)
p0DL3 = np.array(0.4)
pDL3 = np.append(p0DL3,pDL3)
pDL3 = unp.uarray(pDL3,pDL3*0.1)

tDL3 = [0]
tDL3_f = [0]

for i in range (0,len(tDL3_1)):
	th,th_f = avg_and_sem([tDL3_1[i],tDL3_2[i],tDL3_3[i]])
	tDL3.append(th)
	tDL3_f.append(th_f)
	
tDL3=unp.uarray(tDL3,tDL3_f)

paramsLinearDL3, errorsLinearDL3, sigma_y = linregress(noms(tDL3), noms(pDL3))
steigungDL3 = unp.uarray(paramsLinearDL3[0],errorsLinearDL3[0])

SDL3 = VDL/p0DL3*steigungDL3
print('SDL1 = ', SDL1)

#DL4
pDL4,tDL4_1,tDL4_2,tDL4_3 = np.genfromtxt(r'scripts/dataDL4.txt',unpack=True)
p0DL4 = np.array(0.1)
pDL4 = np.append(p0DL4,pDL4)
pDL4 = unp.uarray(pDL4,pDL4*0.1)

tDL4 = [0]
tDL4_f = [0]

for i in range (0,len(tDL4_1)):
	th,th_f = avg_and_sem([tDL4_1[i],tDL4_2[i],tDL4_3[i]])
	tDL4.append(th)
	tDL4_f.append(th_f)
	
tDL4=unp.uarray(tDL4,tDL4_f)

paramsLinearDL4, errorsLinearDL4, sigma_y = linregress(noms(tDL4), noms(pDL4))
steigungDL4 = unp.uarray(paramsLinearDL4[0],errorsLinearDL4[0])

SDL4 = VDL/p0DL4*steigungDL4
print('SDL4 = ', SDL4)

"""
#Plot
#DL1
plt.cla()
plt.clf()
x_plot = np.linspace(-2,80)

plt.errorbar(noms(tDL1), noms(pDL1), xerr=stds(tDL1), yerr=stds(pDL1), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot,x_plot*paramsLinearDL1[0]+paramsLinearDL1[1],'b-',label='Ausgleichsgerade')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$p/\si{\milli\bar}$')
plt.xlim(-2,78)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/DL1.png')

print('DL1 done')

#DL2
plt.cla()
plt.clf()
x_plot = np.linspace(-2,80)

plt.errorbar(noms(tDL2), noms(pDL2), xerr=stds(tDL2), yerr=stds(pDL2), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot,x_plot*paramsLinearDL2[0]+paramsLinearDL2[1],'b-',label='Ausgleichsgerade')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$p/\si{\milli\bar}$')
plt.xlim(-2,80)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/DL2.png')

print('DL2 done')

#DL3
plt.cla()
plt.clf()
x_plot = np.linspace(-5,175)

plt.errorbar(noms(tDL3), noms(pDL3), xerr=stds(tDL3), yerr=stds(pDL3), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot,x_plot*paramsLinearDL3[0]+paramsLinearDL3[1],'b-',label='Ausgleichsgerade')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$p/\si{\milli\bar}$')
plt.xlim(-5,170)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/DL3.png')

print('DL3 done')

#TL4
plt.cla()
plt.clf()
x_plot = np.linspace(-5,200)

plt.errorbar(noms(tDL4), noms(pDL4), xerr=stds(tDL4), yerr=stds(pDL4), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot,x_plot*paramsLinearDL4[0]+paramsLinearDL4[1],'b-',label='Ausgleichsgerade')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$p/\si{\milli\bar}$')
plt.xlim(-5,185)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/DL4.png')

print('DL4 done')

#DL 1-4
plt.cla()
plt.clf()
x_plot = np.linspace(-5,185)

#DL1
plt.errorbar(noms(tDL1), noms(pDL1), xerr=stds(tDL1), yerr=stds(pDL1), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=1 \si{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearDL1[0]+paramsLinearDL1[1],'k-',label='Ausgleichsgeraden')
#DL2
plt.errorbar(noms(tDL2), noms(pDL2), xerr=stds(tDL2), yerr=stds(pDL2), fmt='bx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=0.8 \si{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearDL2[0]+paramsLinearDL2[1],'k-')
#DL3
plt.errorbar(noms(tDL3), noms(pDL3), xerr=stds(tDL3), yerr=stds(pDL3), fmt='cx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=0.4 \si{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearDL3[0]+paramsLinearDL3[1],'k-')
#DL3
plt.errorbar(noms(tDL4), noms(pDL4), xerr=stds(tDL4), yerr=stds(pDL4), fmt='mx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=0.1 \si{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearDL4[0]+paramsLinearDL4[1],'k-')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$p/\si{\milli\bar}$')
plt.xlim(-5,185)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/DL.png')
"""
print('DL done')


