﻿from table2 import makeTable
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

#Matrix
A = np.array([[np.sqrt(2),0,0,0,0,0,0,0,0],
			[0,0,np.sqrt(2),0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,np.sqrt(2)],
			[0,0,0,0,0,0,np.sqrt(2),0,0],
			[1,0,0,1,0,0,1,0,0],
			[0,1,0,0,1,0,0,1,0],
			[0,0,1,0,0,1,0,0,1],
			[1,1,1,0,0,0,0,0,0],
			[0,0,0,1,1,1,0,0,0],
			[0,0,0,0,0,0,1,1,1],
			[0,np.sqrt(2),0,np.sqrt(2),0,0,0,0,0],
			[0,0,np.sqrt(2),0,np.sqrt(2),0,np.sqrt(2),0,0],
			[0,0,0,0,0,np.sqrt(2),0,np.sqrt(2),0],
			[0,np.sqrt(2),0,0,0,np.sqrt(2),0,0,0],
			[np.sqrt(2),0,0,0,np.sqrt(2),0,0,0,np.sqrt(2)],
			[0,0,0,np.sqrt(2),0,0,0,np.sqrt(2),0]])
#print(A)
A_T = np.transpose(A)
#print('A_T:', A_T)

P = np.matmul(A_T,A)
#print('P:', P)
P_inv = np.linalg.inv(P)
#print('P_inv:', P_inv)


#Spektrum
N_E = np.genfromtxt('scripts/RolfBlank/0.Spe',unpack=True)
x = np.linspace(0,len(N_E),len(N_E))
E = 662/113.48*x

plt.cla()
plt.clf()
plt.plot(E,N_E,'r-',label=r'Energiespektrum')
plt.xlabel(r'$E/\si{\kilo\electronvolt}$')
plt.ylabel(r'N')
plt.xlim(0,800)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig('build/Energiespektrum.pdf')

#Würfel 1, Bestimmung von I_0
I0_gerade = unp.uarray(50636,254)/300
print('I0_gerade', I0_gerade)
I0_schraeg1 = unp.uarray(16660,146)/100
I0_schraeg2 = unp.uarray(16417,145)/100
#I0_schraeg = np.mean([I0_schraeg1,I0_schraeg2])
I0_schraeg = avg_and_sem([noms(I0_schraeg1),noms(I0_schraeg2)])
I0_schraeg = unp.uarray(I0_schraeg[0],I0_schraeg[1])
print('I0_schraeg', I0_schraeg)

makeTable([noms([I0_gerade*300,I0_schraeg1*100,I0_schraeg2*100]),stds([I0_gerade*300,I0_schraeg1*100,I0_schraeg2*100]),[300,100,100],noms([I0_gerade,I0_schraeg1,I0_schraeg2]),stds([I0_gerade,I0_schraeg1,I0_schraeg2])], r'\multicolumn{2}{c}{'+r'$N_0$'+r'} & {'+r'$\Delta t_0/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$I_0/\si{\becquerel}$'+r'}','tabReferenzmessung',['S[table-format=5.0]','@{${}\pm{}$}S[table-format=3.0]','S[table-format=3.0]','S[table-format=3.1]','@{${}\pm{}$}S[table-format=1.1]'],["%5.0f","%3.0f","%3.0f","%3.1f","%1.1f"])


#Würfel 2
I2_gerade = unp.uarray(8054,103)/300
I2_schraeg1 = unp.uarray(4864,81)/300
I2_schraeg2 = unp.uarray(8707,106)/300
mu_21 = unp.log(I0_gerade/I2_gerade)/3
mu_22 = unp.log(I0_schraeg1/I2_schraeg1)/(3*np.sqrt(2))
mu_23 = unp.log(I0_schraeg2/I2_schraeg2)/(2*np.sqrt(2))
print('mu_21', mu_21)
print('mu_22', mu_22)
print('mu_23', mu_23)
#mu_2 = np.mean([mu_21,mu_22,mu_23])
mu_2 = avg_and_sem([noms(mu_21),noms(mu_22),noms(mu_23)])
print('mu_2', mu_2)

makeTable([noms([I2_gerade*300,I2_schraeg1*300,I2_schraeg2*300]),stds([I2_gerade*300,I2_schraeg1*300,I2_schraeg2*300]),[300,300,300],noms([I2_gerade,I2_schraeg1,I2_schraeg2]),stds([I2_gerade,I2_schraeg1,I2_schraeg2]),noms([mu_21,mu_22,mu_23]),stds([mu_21,mu_22,mu_23])], r'\multicolumn{2}{c}{'+r'$N$'+r'} & {'+r'$\Delta t/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$I/\si{\becquerel}$'+r'} & \multicolumn{2}{c}{'+r'$\mu/\si{\per\centi\metre}$'+r'}','tabWuerfel2',['S[table-format=4.0]','@{${}\pm{}$}S[table-format=3.0]','S[table-format=3.0]','S[table-format=2.1]','@{${}\pm{}$}S[table-format=1.1]','S[table-format=1.3]','@{${}\pm{}$}S[table-format=1.3]'],["%4.0f","%3.0f","%3.0f","%2.1f","%1.1f","%1.3f","%1.3f"])


#Würfel 3
I3_gerade = unp.uarray(24141,174)/200
I3_schraeg1 = unp.uarray(21479,165)/200
I3_schraeg2 = unp.uarray(24197,176)/200
mu_31 = unp.log(I0_gerade/I3_gerade)/3
mu_32 = unp.log(I0_schraeg1/I3_schraeg1)/(3*np.sqrt(2))
mu_33 = unp.log(I0_schraeg2/I3_schraeg2)/(2*np.sqrt(2))
print('mu_31', mu_31)
print('mu_32', mu_32)
print('mu_33', mu_33)
mu_3 = avg_and_sem([noms(mu_31),noms(mu_32),noms(mu_33)])
print('mu_3', mu_3)

makeTable([noms([I3_gerade*200,I3_schraeg1*200,I3_schraeg2*200]),stds([I3_gerade*200,I3_schraeg1*200,I3_schraeg2*200]),[200,200,200],noms([I3_gerade,I3_schraeg1,I3_schraeg2]),stds([I3_gerade,I3_schraeg1,I3_schraeg2]),noms([mu_31,mu_32,mu_33]),stds([mu_31,mu_32,mu_33])], r'\multicolumn{2}{c}{'+r'$N$'+r'} & {'+r'$\Delta t/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$I/\si{\becquerel}$'+r'} & \multicolumn{2}{c}{'+r'$\mu/\si{\per\centi\metre}$'+r'}','tabWuerfel3',['S[table-format=5.0]','@{${}\pm{}$}S[table-format=3.0]','S[table-format=3.0]','S[table-format=3.1]','@{${}\pm{}$}S[table-format=1.1]','S[table-format=1.3]','@{${}\pm{}$}S[table-format=1.3]'],["%5.0f","%3.0f","%3.0f","%3.1f","%1.1f","%1.3f","%1.3f"])


#Würfel 5
I5_1 = unp.uarray(25250,181)/300
I5_1log = unp.log(I0_schraeg2/I5_1)
mu5_1 = I5_1log/np.sqrt(2)
print('mu5_1',mu5_1)
I5_2 = unp.uarray(42856,234)/300
I5_2log = unp.log(I0_schraeg2/I5_2)
mu5_3 = I5_2log/np.sqrt(2)
print('mu5_3',mu5_3)
I5_3 = unp.uarray(25156,179)/300
I5_3log = unp.log(I0_schraeg2/I5_3)
mu5_9 = I5_3log/np.sqrt(2)
print('mu5_9',mu5_9)
I5_4 = unp.uarray(21835,167)/300
I5_4log = unp.log(I0_schraeg2/I5_4)
mu5_7 = I5_4log/np.sqrt(2)
print('mu5_7',mu5_7)
I5_5 = unp.uarray(8632,107)/300
I5_5log = unp.log(I0_gerade/I5_5)
mu5_4 = I5_5log-mu5_1-mu5_7
print('mu5_4',mu5_4)
I5_6 = unp.uarray(22108,168)/300
I5_6log = unp.log(I0_gerade/I5_6)
I5_7 = unp.uarray(22048,167)/300
I5_7log = unp.log(I0_gerade/I5_7)
mu5_6 = I5_7log-mu5_3-mu5_9
print('mu5_6',mu5_6)
I5_8 = unp.uarray(22296,170)/300
I5_8log = unp.log(I0_gerade/I5_8)
mu5_2 = I5_8log-mu5_1-mu5_3
print('mu5_2',mu5_2)
I5_9 = unp.uarray(21250,165)/300
I5_9log = unp.log(I0_gerade/I5_9)
mu5_5 = I5_9log-mu5_4-mu5_6
print('mu5_5',mu5_5)
I5_10 = unp.uarray(8440,106)/300
I5_10log = unp.log(I0_gerade/I5_10)
mu5_8 = I5_10log-mu5_7-mu5_9
print('mu5_8',mu5_8)
I5_11 = unp.uarray(18139,153)/300
I5_11log = unp.log(I0_schraeg2/I5_11)
I5_12 = unp.uarray(15245,141)/300
I5_12log = unp.log(I0_schraeg1/I5_12)
I5_13 = unp.uarray(17586,150)/300
I5_13log = unp.log(I0_schraeg2/I5_13)
I5_14 = unp.uarray(31775,201)/300
I5_14log = unp.log(I0_schraeg2/I5_14)
I5_15 = unp.uarray(8680,107)/300
I5_15log = unp.log(I0_schraeg1/I5_15)
I5_16 = unp.uarray(9966,114)/300
I5_16log = unp.log(I0_schraeg2/I5_16)

mu_51 = avg_and_sem([noms(mu5_1),noms(mu5_9),noms(mu5_7),noms(mu5_4),noms(mu5_8)])
print('mu_51',mu_51)
mu_52 = avg_and_sem([noms(mu5_3),noms(mu5_6),noms(mu5_2),noms(mu5_5)])
print('mu_52',mu_52)

I5 = unp.uarray([noms(I5_1),noms(I5_2),noms(I5_3),noms(I5_4),noms(I5_5),noms(I5_6),noms(I5_7),noms(I5_8),noms(I5_9),noms(I5_10),noms(I5_11),noms(I5_12),noms(I5_13),noms(I5_14),noms(I5_15),noms(I5_16)],[stds(I5_1),stds(I5_2),stds(I5_3),stds(I5_4),stds(I5_5),stds(I5_6),stds(I5_7),stds(I5_8),stds(I5_9),stds(I5_10),stds(I5_11),stds(I5_12),stds(I5_13),stds(I5_14),stds(I5_15),stds(I5_16)])
I5log = unp.uarray([noms(I5_1log),noms(I5_2log),noms(I5_3log),noms(I5_4log),noms(I5_5log),noms(I5_6log),noms(I5_7log),noms(I5_8log),noms(I5_9log),noms(I5_10log),noms(I5_11log),noms(I5_12log),noms(I5_13log),noms(I5_14log),noms(I5_15log),noms(I5_16log)],[stds(I5_1log),stds(I5_2log),stds(I5_3log),stds(I5_4log),stds(I5_5log),stds(I5_6log),stds(I5_7log),stds(I5_8log),stds(I5_9log),stds(I5_10log),stds(I5_11log),stds(I5_12log),stds(I5_13log),stds(I5_14log),stds(I5_15log),stds(I5_16log)])

M_5 = np.matmul(P_inv,A_T)
#print('M_5', M_5)
mu_5 = np.matmul(M_5,noms(I5log))
mu_5 = unp.uarray(mu_5,mu_5)*0
for i in range(M_5.shape[0]):
	for j in range(M_5.shape[1]):
		mu_5[i] += M_5[i][j]*I5log[j]
print('mu_5', mu_5)
mu_51 = avg_and_sem(noms(mu_5[mu_5>0.5]))
print('mu_51', mu_51)
mu_52 = avg_and_sem(noms(mu_5[mu_5<0.5]))
print('mu_52', mu_52)

makeTable([noms(I5*300),stds(I5*300),[300,300,300,300,300,300,300,300,300,300,300,300,300,300,300,300],noms(I5),stds(I5),noms(I5log),stds(I5log)], r'\multicolumn{2}{c}{'+r'$N$'+r'} & {'+r'$\Delta t/\si{\second}$'+r'} & \multicolumn{2}{c}{'+r'$I/\si{\becquerel}$'+r'} & \multicolumn{2}{c}{'+r'$\log\left(\frac{I_0}{I}\right)$'+r'}','tabWuerfel5',['S[table-format=5.0]','@{${}\pm{}$}S[table-format=3.0]','S[table-format=3.0]','S[table-format=3.1]','@{${}\pm{}$}S[table-format=1.1]','S[table-format=1.3]','@{${}\pm{}$}S[table-format=1.3]'],["%5.0f","%3.0f","%3.0f","%3.1f","%1.1f","%1.3f","%1.3f"])


#ohne die ersten 4 Werte
A = A[4:]
A_T = np.transpose(A)
P = np.matmul(A_T,A)
P_inv = np.linalg.inv(P)
I5log = I5log[4:]
M_5 = np.matmul(P_inv,A_T)
mu_5 = np.matmul(M_5,noms(I5log))
mu_5 = unp.uarray(mu_5,mu_5)*0
for i in range(M_5.shape[0]):
	for j in range(M_5.shape[1]):
		mu_5[i] += M_5[i][j]*I5log[j]
print('mu_5', mu_5)
mu_51 = avg_and_sem(noms(mu_5[mu_5>0.5]))
print('mu_51', mu_51)
mu_52 = avg_and_sem(noms(mu_5[mu_5<0.5]))
print('mu_52', mu_52)
