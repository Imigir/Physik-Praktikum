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
p0TL1 = np.array(2)
pTL1 = np.append(p0TL1,pTL1)
pTL1 = unp.uarray(pTL1,pTL1*0.1)

tTL1 = [0]
tTL1_f = [0]

for i in range (0,len(tTL1_1)):
	th,th_f = avg_and_sem([tTL1_1[i],tTL1_2[i],tTL1_3[i]])
	tTL1.append(th)
	tTL1_f.append(th_f)
	
tTL1=unp.uarray(tTL1,tTL1_f)
#print('t',tTL1)

paramsLinearTL1, errorsLinearTL1, sigma_y = linregress(noms(tTL1), noms(pTL1))
steigungTL1 = unp.uarray(paramsLinearTL1[0],errorsLinearTL1[0])
achsenAbschnittTL1 = unp.uarray(paramsLinearTL1[1], errorsLinearTL1[1])
paramsLinearTL1_2, errorsLinearTL1_2, sigma_y = linregress(noms(tTL1)[3:], noms(pTL1)[3:])
steigungTL1_2 = unp.uarray(paramsLinearTL1_2[0],errorsLinearTL1_2[0])
achsenAbschnittTL1_2 = unp.uarray(paramsLinearTL1_2[1], errorsLinearTL1_2[1])

print('steigungTL1 =', steigungTL1)
print('achsenAbschnittTL1 =', achsenAbschnittTL1)
print('steigungTL1_2 =', steigungTL1_2)
print('achsenAbschnittTL1_2 =', achsenAbschnittTL1_2)

STL1 = VTL/p0TL1*steigungTL1
print('STL1 =', STL1)
STL1_2 = VTL/p0TL1*steigungTL1_2
print('STL1_2 =', STL1_2)

makeTable([noms(pTL1)[1:], stds(pTL1)[1:], tTL1_1, tTL1_2, tTL1_3, noms(tTL1)[1:], stds(tTL1)[1:]], r'\multicolumn{2}{c}{'+r'$p/10^{-4}\si{\milli\bar}$'+r'} & {'+r'$t_1/\si{\second}$'+r'} & {'+r'$t_2/\si{\second}$'+r'} & {'+r'$t_3/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$\bar{t}/\si{\second}$'+r'}', 'tabTL1', ['S[table-format=2.0]', '@{${}\pm{}$} S[table-format=1.1]','S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', ' @{${}\pm{}$} S[table-format=1.2]'], ["%2.0f", "%1.1f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%1.2f"])

#TL2
pTL2,tTL2_1,tTL2_2,tTL2_3 = np.genfromtxt(r'scripts/dataTL2.txt',unpack=True)
p0TL2 = np.array(1.4)
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
steigungTL2 = unp.uarray(paramsLinearTL2[0],errorsLinearTL2[0])
achsenAbschnittTL2 = unp.uarray(paramsLinearTL2[1], errorsLinearTL2[1])
paramsLinearTL2_2, errorsLinearTL2_2, sigma_y = linregress(noms(tTL2)[3:], noms(pTL2)[3:])
steigungTL2_2 = unp.uarray(paramsLinearTL2_2[0],errorsLinearTL2_2[0])
achsenAbschnittTL2_2 = unp.uarray(paramsLinearTL2_2[1], errorsLinearTL2_2[1])

print('steigungTL2 =', steigungTL2)
print('achsenAbschnittTL2 =', achsenAbschnittTL2)
print('steigungTL2_2 =', steigungTL2_2)
print('achsenAbschnittTL2_2 =', achsenAbschnittTL2_2)


STL2 = VTL/p0TL2*steigungTL2
print('STL2 =', STL2)
STL2_2 = VTL/p0TL2*steigungTL2_2
print('STL2_2 =', STL2_2)

makeTable([noms(pTL2)[1:], stds(pTL2)[1:], tTL2_1, tTL2_2, tTL2_3, noms(tTL2)[1:], stds(tTL2)[1:]], r'\multicolumn{2}{c}{'+r'$p/10^{-4}\si{\milli\bar}$'+r'} & {'+r'$t_1/\si{\second}$'+r'} & {'+r'$t_2/\si{\second}$'+r'} & {'+r'$t_3/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$\bar{t}/\si{\second}$'+r'}', 'tabTL2', ['S[table-format=2.0]', ' @{${}\pm{}$} S[table-format=1.1]','S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', ' @{${}\pm{}$} S[table-format=1.2]'], ["%2.0f", "%1.1f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%1.2f"])

#TL3
pTL3,tTL3_1,tTL3_2,tTL3_3 = np.genfromtxt(r'scripts/dataTL3.txt',unpack=True)
p0TL3 = np.array(1)
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
steigungTL3 = unp.uarray(paramsLinearTL3[0],errorsLinearTL3[0])
achsenAbschnittTL3 = unp.uarray(paramsLinearTL3[1], errorsLinearTL3[1])
paramsLinearTL3_2, errorsLinearTL3_2, sigma_y = linregress(noms(tTL3)[3:], noms(pTL3)[3:])
steigungTL3_2 = unp.uarray(paramsLinearTL3_2[0],errorsLinearTL3_2[0])
achsenAbschnittTL3_2 = unp.uarray(paramsLinearTL3_2[1], errorsLinearTL3_2[1])

print('steigungTL3 =', steigungTL3)
print('achsenAbschnittTL3 =', achsenAbschnittTL3)
print('steigungTL3_2 =', steigungTL3_2)
print('achsenAbschnittTL3_2 =', achsenAbschnittTL3_2)

STL3 = VTL/p0TL3*steigungTL3
print('STL3 = ', STL3)
STL3_2 = VTL/p0TL3*steigungTL3_2
print('STL3_2 = ', STL3_2)

makeTable([noms(pTL3)[1:], stds(pTL3)[1:], tTL3_1, tTL3_2, tTL3_3, noms(tTL3)[1:], stds(tTL3)[1:]], r'\multicolumn{2}{c}{'+r'$p/10^{-4}\si{\milli\bar}$'+r'} & {'+r'$t_1/\si{\second}$'+r'} & {'+r'$t_2/\si{\second}$'+r'} & {'+r'$t_3/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$\bar{t}/\si{\second}$'+r'}', 'tabTL3', ['S[table-format=2.0]', ' @{${}\pm{}$} S[table-format=1.1]','S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', ' @{${}\pm{}$} S[table-format=1.2]'], ["%2.0f", "%1.1f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%1.2f"])

#TL4
pTL4,tTL4_1,tTL4_2,tTL4_3 = np.genfromtxt(r'scripts/dataTL4.txt',unpack=True)
p0TL4 = np.array(0.5)
pTL4 = np.append(p0TL4,pTL4)
pTL4 = unp.uarray(pTL4,pTL4*0.1)

tTL4 = [0]
tTL4_f = [0]

for i in range (0,len(tTL4_1)):
	th,th_f = avg_and_sem([tTL4_1[i],tTL4_2[i],tTL4_3[i]])
	tTL4.append(th)
	tTL4_f.append(th_f)
	
tTL4=unp.uarray(tTL4,tTL4_f)

paramsLinearTL4, errorsLinearTL4, sigma_y = linregress(noms(tTL4)[:10], noms(pTL4)[:10])
steigungTL4 = unp.uarray(paramsLinearTL4[0],errorsLinearTL4[0])
achsenAbschnittTL4 = unp.uarray(paramsLinearTL4[1], errorsLinearTL4[1])

print('steigungTL4 =', steigungTL4)
print('achsenAbschnittTL4 =', achsenAbschnittTL4)

STL4 = VTL/p0TL4*steigungTL4
print('STL4 = ', STL4)

makeTable([noms(pTL4)[1:], stds(pTL4)[1:], tTL4_1, tTL4_2, tTL4_3, noms(tTL4)[1:], stds(tTL4)[1:]], r'\multicolumn{2}{c}{'+r'$p/10^{-4}\si{\milli\bar}$'+r'} & {'+r'$t_1/\si{\second}$'+r'} & {'+r'$t_2/\si{\second}$'+r'} & {'+r'$t_3/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$\bar{t}/\si{\second}$'+r'}', 'tabTL4', ['S[table-format=2.0]', ' @{${}\pm{}$} S[table-format=1.1]','S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', ' @{${}\pm{}$} S[table-format=1.2]'], ["%2.0f", "%1.1f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%1.2f"])


#Plot
#TL1
plt.cla()
plt.clf()
x_plot = np.linspace(-2,15)

plt.errorbar(noms(tTL1), noms(pTL1), xerr=stds(tTL1), yerr=stds(pTL1), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot,x_plot*paramsLinearTL1[0]+paramsLinearTL1[1],'b-',label='Ausgleichsgerade 1')
plt.plot(x_plot,x_plot*paramsLinearTL1_2[0]+paramsLinearTL1_2[1],'c-',label='Ausgleichsgerade 2')

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
plt.plot(x_plot,x_plot*paramsLinearTL2[0]+paramsLinearTL2[1],'b-',label='Ausgleichsgerade 1')
plt.plot(x_plot,x_plot*paramsLinearTL2_2[0]+paramsLinearTL2_2[1],'c-',label='Ausgleichsgerade 2')

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
plt.plot(x_plot,x_plot*paramsLinearTL3[0]+paramsLinearTL3[1],'b-',label='Ausgleichsgerade 1')
plt.plot(x_plot,x_plot*paramsLinearTL3_2[0]+paramsLinearTL3_2[1],'c-',label='Ausgleichsgerade 2')

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

plt.errorbar(noms(tTL4)[:10], noms(pTL4)[:10], xerr=stds(tTL4)[:10], yerr=stds(pTL4)[:10], fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.errorbar(noms(tTL4)[10], noms(pTL4)[10], xerr=stds(tTL4)[10], yerr=stds(pTL4)[10], color='grey', marker='x', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Ungenutzt')
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
plt.errorbar(noms(tTL1), noms(pTL1), xerr=stds(tTL1), yerr=stds(pTL1), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=\SI{2e-4}{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearTL1[0]+paramsLinearTL1[1],'k-',label='Ausgleichsgeraden')
#TL2
plt.errorbar(noms(tTL2), noms(pTL2), xerr=stds(tTL2), yerr=stds(pTL2), fmt='bx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=\SI{1.4e-4}{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearTL2[0]+paramsLinearTL2[1],'k-')
#TL3
plt.errorbar(noms(tTL3), noms(pTL3), xerr=stds(tTL3), yerr=stds(pTL3), fmt='cx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=\SI{1e-4}{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearTL3[0]+paramsLinearTL3[1],'k-')
#TL3
plt.errorbar(noms(tTL4), noms(pTL4), xerr=stds(tTL4), yerr=stds(pTL4), fmt='mx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=\SI{0.5e-4}{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearTL4[0]+paramsLinearTL4[1],'k-')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$p/10^{-4}\si{\milli\bar}$')
plt.xlim(-1,31)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/TL.png')

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

makeTable([noms(pDL1)[1:], stds(pDL1)[1:], tDL1_1, tDL1_2, tDL1_3, tDL1_4, noms(tDL1)[1:], stds(tDL1)[1:]], r'\multicolumn{2}{c}{'+r'$p\si{\milli\bar}$'+r'} & {'+r'$t_1/\si{\second}$'+r'} & {'+r'$t_2/\si{\second}$'+r'} & {'+r'$t_3/\si{\second}$'+r'} & {'+r'$t_4/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$\bar{t}/\si{\second}$'+r'}', 'tabDL1', ['S[table-format=2.0]', ' @{${}\pm{}$} S[table-format=1.1]','S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', ' @{${}\pm{}$} S[table-format=1.2]'], ["%2.0f", "%1.1f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%1.2f"])

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

paramsLinearDL2, errorsLinearDL2, sigma_y = linregress(noms(tDL2)[:5], noms(pDL2)[:5])
steigungDL2 = unp.uarray(paramsLinearDL2[0],errorsLinearDL2[0])

SDL2 = VDL/p0DL2*steigungDL2
print('SDL2 = ', SDL2)

makeTable([noms(pDL2)[1:], stds(pDL2)[1:], tDL2_1, tDL2_2, tDL2_3, noms(tDL2)[1:], stds(tDL2)[1:]], r'\multicolumn{2}{c}{'+r'$p/\si{\milli\bar}$'+r'} & {'+r'$t_1/\si{\second}$'+r'} & {'+r'$t_2/\si{\second}$'+r'} & {'+r'$t_3/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$\bar{t}/\si{\second}$'+r'}', 'tabDL2', ['S[table-format=2.0]', ' @{${}\pm{}$} S[table-format=1.1]','S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', ' @{${}\pm{}$} S[table-format=1.2]'], ["%2.0f", "%1.1f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%1.2f"])

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
print('SDL3 = ', SDL3)

makeTable([noms(pDL3)[1:], stds(pDL3)[1:], tDL3_1, tDL3_2, tDL3_3, noms(tDL3)[1:], stds(tDL3)[1:]], r'\multicolumn{2}{c}{'+r'$p/\si{\milli\bar}$'+r'} & {'+r'$t_1/\si{\second}$'+r'} & {'+r'$t_2/\si{\second}$'+r'} & {'+r'$t_3/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$\bar{t}/\si{\second}$'+r'}', 'tabDL3', ['S[table-format=1.1]', ' @{${}\pm{}$} S[table-format=1.1]','S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', ' @{${}\pm{}$} S[table-format=1.2]'], ["%1.1f", "%1.1f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%1.2f"])

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

makeTable([noms(pDL4)[1:], stds(pDL4)[1:], tDL4_1, tDL4_2, tDL4_3, noms(tDL4)[1:], stds(tDL4)[1:]], r'\multicolumn{2}{c}{'+r'$p/\si{\milli\bar}$'+r'} & {'+r'$t_1/\si{\second}$'+r'} & {'+r'$t_2/\si{\second}$'+r'} & {'+r'$t_3/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$\bar{t}/\si{\second}$'+r'}', 'tabDL4', ['S[table-format=1.1]', ' @{${}\pm{}$} S[table-format=1.1]','S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', ' @{${}\pm{}$} S[table-format=1.2]'], ["%1.1f", "%1.1f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%1.2f"])

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

plt.errorbar(noms(tDL2)[:5], noms(pDL2)[:5], xerr=stds(tDL2)[:5], yerr=stds(pDL2)[:5], fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.errorbar(noms(tDL2)[5], noms(pDL2)[5], xerr=stds(tDL2)[5], yerr=stds(pDL2)[5], color='grey', marker='x', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Ungenutzt')
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
plt.errorbar(noms(tDL1), noms(pDL1), xerr=stds(tDL1), yerr=stds(pDL1), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=\SI{1}{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearDL1[0]+paramsLinearDL1[1],'k-',label='Ausgleichsgeraden')
#DL2
plt.errorbar(noms(tDL2)[:5], noms(pDL2)[:5], xerr=stds(tDL2)[:5], yerr=stds(pDL2)[:5], fmt='bx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=\SI{0.8}{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearDL2[0]+paramsLinearDL2[1],'k-')
#DL3
plt.errorbar(noms(tDL3), noms(pDL3), xerr=stds(tDL3), yerr=stds(pDL3), fmt='cx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=\SI{0.4}{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearDL3[0]+paramsLinearDL3[1],'k-')
#DL3
plt.errorbar(noms(tDL4), noms(pDL4), xerr=stds(tDL4), yerr=stds(pDL4), fmt='mx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label=r'$p_0=\SI{0.1}{\milli\bar}$')
plt.plot(x_plot,x_plot*paramsLinearDL4[0]+paramsLinearDL4[1],'k-')
plt.errorbar(noms(tDL2)[5], noms(pDL2)[5], xerr=stds(tDL2)[5], yerr=stds(pDL2)[5], color='grey', marker='x', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Ungenutzt')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$p/\si{\milli\bar}$')
plt.xlim(-5,185)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/DL.png')
"""
print('DL done')


#TS
VTS = unp.uarray(10.298,0.92)

pTS,tTS_1,tTS_2,tTS_3,tTS_4,tTS_5,tTS_6 = np.genfromtxt(r'scripts/dataTS1.txt',unpack=True)
p0TS = np.array(500)
pTS = np.append(p0TS,pTS)
pTS = unp.uarray(pTS,pTS*0.2)
pTS_end = 1.2
pTS_end = unp.uarray(pTS_end,pTS_end*0.2)

tTS = [0]
tTS_f = [0]

for i in range (0,len(tTS_1)):
	th,th_f = avg_and_sem([tTS_1[i],tTS_2[i],tTS_3[i],tTS_4[i],tTS_5[i],tTS_6[i]])
	tTS.append(th)
	tTS_f.append(th_f)

tTS=unp.uarray(tTS,tTS_f)

#pTS_log = np.log(noms(pTS_h))
#pTS_log_err = 1/noms(pTS_h)*stds(pTS_h)
#pTS_log_err = [np.log(noms(pTS_h)+stds(pTS_h))-np.log(noms(pTS_h)), np.log(noms(pTS_h))-np.log(noms(pTS_h)-stds(pTS_h))]
#pTS_log = unp.uarray(pTS_log,pTS_log_err)
pTS_h = (pTS-pTS_end)/(pTS[0]-pTS_end)
pTS_log = unp.log(pTS_h)
#print('pTS_log', pTS_log)

paramsLinearTS1, errorsLinearTS1, sigma_y = linregress(noms(tTS)[0:4], noms(pTS_log)[0:4])
steigungTS1 = unp.uarray(paramsLinearTS1[0],errorsLinearTS1[0])
paramsLinearTS2, errorsLinearTS2, sigma_y = linregress(noms(tTS)[4:8], noms(pTS_log)[4:8])
steigungTS2 = unp.uarray(paramsLinearTS2[0],errorsLinearTS2[0])
paramsLinearTS3, errorsLinearTS3, sigma_y = linregress(noms(tTS)[6:10], noms(pTS_log)[6:10])
steigungTS3 = unp.uarray(paramsLinearTS3[0],errorsLinearTS3[0])

STS1 = steigungTS1*(-VTS) 
STS2 = steigungTS2*(-VTS) 
STS3 = steigungTS3*(-VTS) 
print('STS1 = ', STS1)
print('STS2 = ', STS2)
print('STS3 = ', STS3)

makeTable([noms(pTS)[1:], stds(pTS)[1:], noms(pTS_log)[1:], stds(pTS_log)[1:], tTS_1, tTS_2, tTS_3, tTS_4, tTS_5, tTS_6, noms(tTS)[1:], stds(tTS)[1:]], r'\multicolumn{2}{c}{'+r'$p/10^{-5}\si{\milli\bar}$'+r'} & \multicolumn{2}{c}{'+r'$\log\left(\frac{p-p_e}{p_0-p_e}\right)$'+r'} & {'+r'$t_1/\si{\second}$'+r'} & {'+r'$t_2/\si{\second}$'+r'} & {'+r'$t_3/\si{\second}$'+r'} & {'+r'$t_4/\si{\second}$'+r'} & {'+r'$t_5/\si{\second}$'+r'} & {'+r'$t_6/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$\bar{t}/\si{\second}$'+r'}', 'tabTS', ['S[table-format=3.1]', ' @{${}\pm{}$} S[table-format=2.1]', 'S[table-format=2.1]', ' @{${}\pm{}$} S[table-format=1.1]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', 'S[table-format=2.2]', ' @{${}\pm{}$} S[table-format=1.2]'], ["%3.1f", "%2.1f", "%2.1f", "%1.1f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%2.2f", "%1.2f"])


#Plot
#TSE
plt.cla()
plt.clf()
x_plot = np.linspace(-2,15)

plt.errorbar(noms(tTS), noms(pTS), xerr=stds(tTS), yerr=stds(pTS), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$p/10^{-5}\si{\milli\bar}$')
plt.xlim(-1,16)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/TSE.png')
print('TSE done')

#TSL
plt.cla()
plt.clf()
x_plot = np.linspace(-1,16)

plt.errorbar(noms(tTS), noms(pTS_log), xerr=stds(tTS), yerr=stds(pTS_log), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot,x_plot*paramsLinearTS1[0]+paramsLinearTS1[1],'b-',label='Ausgleichsgerade 1')
plt.plot(x_plot,x_plot*paramsLinearTS2[0]+paramsLinearTS2[1],'c-',label='Ausgleichsgerade 2')
plt.plot(x_plot,x_plot*paramsLinearTS3[0]+paramsLinearTS3[1],'m-',label='Ausgleichsgerade 3')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$\log\left(\frac{p-p_e}{p_0-p_e}\right)$')
plt.xlim(-1,16)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/TSL.png')
print('TSL done')

print('TS done')


#DS
VDS = unp.uarray(11.14,1.031)

pDS,tDS_1,tDS_2,tDS_3,tDS_4,tDS_5 = np.genfromtxt(r'scripts/dataDS1.txt',unpack=True)
p0DS = np.array(1013)
pDS = np.append(p0DS,pDS)
pDS = unp.uarray(pDS,pDS*0.2)
pDS_end = 0.02
pDS_end = unp.uarray(pDS_end,pDS_end*0.2)

tDS = [0]
tDS_f = [0]

for i in range (0,len(tDS_1)):
	th,th_f = avg_and_sem([tDS_1[i],tDS_2[i],tDS_3[i],tDS_4[i],tDS_5[i]])
	tDS.append(th)
	tDS_f.append(th_f)

tDS=unp.uarray(tDS,tDS_f)

pDS_h = (pDS-pDS_end)/(pDS[0]-pDS_end)
pDS_log = unp.log(pDS_h)
#print('pDS_log',pDS_log)

paramsLinearDS1, errorsLinearDS1, sigma_y = linregress(noms(tDS)[0:11], noms(pDS_log)[0:11])
steigungDS1 = unp.uarray(paramsLinearDS1[0],errorsLinearDS1[0])
paramsLinearDS2, errorsLinearDS2, sigma_y = linregress(noms(tDS)[11:17], noms(pDS_log)[11:17])
steigungDS2 = unp.uarray(paramsLinearDS2[0],errorsLinearDS2[0])

SDS1 = steigungDS1*(-VDS) 
SDS2 = steigungDS2*(-VDS) 
print('SDS1 = ', SDS1)
print('SDS2 = ', SDS2)

makeTable([noms(pDS)[1:], stds(pDS)[1:], noms(pDS_log)[1:], stds(pDS_log)[1:], tDS_1, tDS_2, tDS_3, tDS_4, tDS_5, noms(tDS)[1:], stds(tDS)[1:]], r'\multicolumn{2}{c}{'+r'$p/\si{\milli\bar}$'+r'} & \multicolumn{2}{c}{'+r'$\log\left(\frac{p-p_e}{p_0-p_e}\right)$'+r'} & {'+r'$t_1/\si{\second}$'+r'} & {'+r'$t_2/\si{\second}$'+r'} & {'+r'$t_3/\si{\second}$'+r'} & {'+r'$t_4/\si{\second}$'+r'} & {'+r'$t_5/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$\bar{t}/\si{\second}$'+r'}', 'tabDS', ['S[table-format=3.2]', ' @{${}\pm{}$} S[table-format=2.2]', 'S[table-format=3.1]', ' @{${}\pm{}$} S[table-format=1.1]', 'S[table-format=3.2]', 'S[table-format=3.2]', 'S[table-format=3.2]', 'S[table-format=3.2]', 'S[table-format=3.2]', 'S[table-format=3.2]', '@{${}\pm{}$} S[table-format=1.2]'], ["%3.2f", "%2.2f", "%3.1f", "%1.1f", "%3.2f", "%3.2f", "%3.2f", "%3.2f", "%3.2f", "%3.2f", "%1.2f"])


#Plot
#DSE
plt.cla()
plt.clf()

plt.errorbar(noms(tDS), noms(pDS), xerr=stds(tDS), yerr=stds(pDS), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$p/\si{\milli\bar}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/DSE.png')
print('DSE done')

#DSL
plt.cla()
plt.clf()
x_plot = np.linspace(-5,135)

plt.errorbar(noms(tDS), noms(pDS_log), xerr=stds(tDS), yerr=stds(pDS_log), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Messwerte')
plt.plot(x_plot,x_plot*paramsLinearDS1[0]+paramsLinearDS1[1],'b-',label='Ausgleichsgerade 1')
plt.plot(x_plot,x_plot*paramsLinearDS2[0]+paramsLinearDS2[1],'c-',label='Ausgleichsgerade 2')

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$\log\left(\frac{p-p_e}{p_0-p_e}\right)$')
plt.xlim(-5,135)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/DSL.png')
print('DSL done')

print('DS done')


#Ges

#GesT
plt.cla()
plt.clf()

#plt.errorbar([noms(pTL1)[0]*10,noms(pTL1)[-1]*10], [noms(STL1),noms(STL1)], xerr=[stds(pTL1)[0]*10,stds(pTL1)[-1]*10], yerr=[stds(STL1),stds(STL1)], fmt='b-', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TL1')
#plt.errorbar([noms(pTL2)[0]*10,noms(pTL2)[-1]*10], [noms(STL2),noms(STL2)], xerr=[stds(pTL2)[0]*10,stds(pTL2)[-1]*10], yerr=[stds(STL2),stds(STL2)], fmt='c-', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TL2')
#plt.errorbar([noms(pTL3)[0]*10,noms(pTL3)[-1]*10], [noms(STL3),noms(STL3)], xerr=[stds(pTL3)[0]*10,stds(pTL3)[-1]*10], yerr=[stds(STL3),stds(STL3)], fmt='m-', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TL3')
#plt.errorbar([noms(pTL4)[0]*10,noms(pTL4)[-1]*10], [noms(STL4),noms(STL4)], xerr=[stds(pTL4)[0]*10,stds(pTL4)[-1]*10], yerr=[stds(STL1),stds(STL1)], fmt='r-', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TL4')
plt.errorbar([noms(pTS)[0],noms(pTS)[3]], [noms(STS1),noms(STS1)], xerr=[stds(pTS)[0],stds(pTS)[3]], yerr=[stds(STS1),stds(STS1)], fmt='b--', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TS1')
plt.errorbar([noms(pTS)[4],noms(pTS)[7]], [noms(STS2),noms(STS2)], xerr=[stds(pTS)[4],stds(pTS)[7]], yerr=[stds(STS2),stds(STS2)], fmt='c--', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TS2')
plt.errorbar([noms(pTS)[6],noms(pTS)[9]], [noms(STS3),noms(STS3)], xerr=[stds(pTS)[6],stds(pTS)[9]], yerr=[stds(STS3),stds(STS3)], fmt='m--', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TS3')

plt.errorbar(noms(pTL1)[0]*10, noms(STL1), xerr=stds(pTL1)[0]*10, yerr=stds(STL1), fmt='bx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TL1.1')
plt.errorbar(noms(pTL1)[0]*10, noms(STL1_2), xerr=stds(pTL1)[0]*10, yerr=stds(STL1_2), fmt='bo', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TL1.2')
plt.errorbar(noms(pTL2)[0]*10, noms(STL2), xerr=stds(pTL2)[0]*10, yerr=stds(STL2), fmt='cx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TL2.1')
plt.errorbar(noms(pTL2)[0]*10, noms(STL2_2), xerr=stds(pTL2)[0]*10, yerr=stds(STL2_2), fmt='co', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TL2.2')
plt.errorbar(noms(pTL3)[0]*10, noms(STL3), xerr=stds(pTL3)[0]*10, yerr=stds(STL3), fmt='mx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TL3.1')
plt.errorbar(noms(pTL3)[0]*10, noms(STL3_2), xerr=stds(pTL3)[0]*10, yerr=stds(STL3_2), fmt='mo', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TL3.2')
plt.errorbar(noms(pTL4)[0]*10, noms(STL4), xerr=stds(pTL4)[0]*10, yerr=stds(STL1), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TL4')
#plt.errorbar(noms(pTS)[0], noms(STS1), xerr=stds(pTS)[0], yerr=stds(STS1), fmt='ro', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TS1')
#plt.errorbar(noms(pTS)[3], noms(STS2), xerr=stds(pTS)[3], yerr=stds(STS2), fmt='co', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TS2')
#plt.errorbar(noms(pTS)[6], noms(STS3), xerr=stds(pTS)[6], yerr=stds(STS3), fmt='mo', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TS3')

plt.xlabel(r'$p/10^{-5}\si{\milli\bar}$')
plt.ylabel(r'$S/\si{\litre\per\second}$')
plt.xlim(-5,50)
plt.legend(loc='best', fancybox=True, framealpha=1)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/TGes.png')
print('TGes done')

#GesD
plt.cla()
plt.clf()

#plt.errorbar([noms(pDL1)[0],noms(pDL1)[-1]], [noms(SDL1),noms(SDL1)], xerr=[stds(pDL1)[0],stds(pDL1)[-1]], yerr=[stds(SDL1),stds(SDL1)], fmt='b-', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='DL1')
#plt.errorbar([noms(pDL2)[0],noms(pDL2)[-1]], [noms(SDL2),noms(SDL2)], xerr=[stds(pDL2)[0],stds(pDL2)[-1]], yerr=[stds(SDL2),stds(SDL2)], fmt='c-', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='DL2')
#plt.errorbar([noms(pDL3)[0],noms(pDL3)[-1]], [noms(SDL3),noms(SDL3)], xerr=[stds(pDL3)[0],stds(pDL3)[-1]], yerr=[stds(SDL3),stds(SDL3)], fmt='m-', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='DL3')
#plt.errorbar([noms(pDL4)[0],noms(pDL4)[-1]], [noms(SDL4),noms(SDL4)], xerr=[stds(pDL4)[0],stds(pDL4)[-1]], yerr=[stds(SDL1),stds(SDL1)], fmt='r-', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='DL4')
plt.errorbar([noms(pDS)[5],noms(pDS)[10]], [noms(SDS1),noms(SDS1)], xerr=[stds(pDS)[5],stds(pDS)[10]], yerr=[stds(SDS1),stds(SDS1)], fmt='b--', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='DS1')
plt.errorbar([noms(pDS)[11],noms(pDS)[16]], [noms(SDS2),noms(SDS2)], xerr=[stds(pDS)[11],stds(pDS)[16]], yerr=[stds(SDS2),stds(SDS2)], fmt='c--', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='DS2')

plt.errorbar(noms(pDL1)[0], noms(SDL1), xerr=stds(pDL1)[0], yerr=stds(SDL1), fmt='bx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='DL1')
plt.errorbar(noms(pDL2)[0], noms(SDL2), xerr=stds(pDL2)[0], yerr=stds(SDL2), fmt='cx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='DL2')
plt.errorbar(noms(pDL3)[0], noms(SDL3), xerr=stds(pDL3)[0], yerr=stds(SDL3), fmt='mx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='DL3')
plt.errorbar(noms(pDL4)[0], noms(SDL4), xerr=stds(pDL4)[0], yerr=stds(SDL4), fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='DL4')
#plt.errorbar(noms(pTS)[0], noms(STS1), xerr=stds(pTS)[0], yerr=stds(STS1), fmt='ro', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TS1')
#plt.errorbar(noms(pTS)[3], noms(STS2), xerr=stds(pTS)[3], yerr=stds(STS2), fmt='co', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TS2')
#plt.errorbar(noms(pTS)[6], noms(STS3), xerr=stds(pTS)[6], yerr=stds(STS3), fmt='mo', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='TS3')

plt.xlabel(r'$p/\si{\milli\bar}$')
plt.ylabel(r'$S/\si{\litre\per\second}$')
plt.xlim(-0.5,3)
plt.legend(loc='best', fancybox=True, framealpha=1)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('content/images/DGes.png')
print('DGes done')

print('Ges done')
