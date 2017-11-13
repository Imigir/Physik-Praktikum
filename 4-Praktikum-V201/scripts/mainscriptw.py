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
# unp.uarray(*avg_and_sem(values))
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

cw = 4.18
ml = 193.25
Tm = 46.6
mx = 270.16
Tx = 21.1
mw = 691.3
my = mw-ml
Ty = 83

makeNewTable([[Tx], [Ty], [mx], [my]],r'{$T_\text{x}/\si[per-mode=reciprocal]{\kelvin}$}&{$T_\text{y}/\si[per-mode=reciprocal]{\kelvin}$}&{$m_\text{x}/\si[per-mode=reciprocal]{\gram}$}&{$m_\text{y}/\si[per-mode=reciprocal]{\gram}$}','tab1', ['S[table-format=3.2] ','S[table-format=3.2]',r'S[table-format=3.2]','S[table-format=3.2]'], [r'{:5.2f}',r'{:5.2f}',r'{:5.2f}',r'{:5.2f}'])


#andere Werte
#Tx = 295.95
#mx = 277.89
#Ty = 365.55
#my = 209.90
#Tm = 322.65 

cgmg = (cw*my*(Ty-Tm)-cw*mx*(Tm-Tx))/(Tm-Tx)
#cgmg = 0.2*500

def c(a):
	mk = a[0]
	Tk = a[1]
	mw = a[2]
	Tw = a[3]
	Tm = a[4]
	return ((cw*(mw-ml) + cgmg)*(Tm-Tw))/(mk*(Tk-Tm))
	
def CV(b):
	c = b[0]
	M = b[1]
	a = b[2]*10**(-6)
	k = b[3]*10**9/1000
	p = b[4]*1000
	Tm = b[5]+273.15
	return (c*M-9*a**2*k*M/p*Tm)
	


mcu = 235.55
cu1 = [mcu, 84, 712.92, 22, 24.9]
cu2 = [mcu, 83.9, 716.42, 22.6, 24.7]
cu3 = [mcu, 85, 676.5, 22.1, 24.4]

mpb = 541.89
pb1 = [mpb, 83, 691.58, 23.0, 25.4]
pb2 = [mpb, 85.3, 693.24, 21.6, 24.1]
pb3 = [mpb, 93, 713.6, 21.3, 24.2]

mgr = 106.43
gr1 = [mgr, 83.3, 703.94, 21.6, 24.5]
gr2 = [mgr, 84.0, 669.04, 21.6, 24.1]
gr3 = [mgr, 84.4, 697.62, 21.6, 24.0]
	
ccu1 = c(cu1) 
ccu2 = c(cu2)
ccu3 = c(cu3)
ccum = unp.uarray(*avg_and_sem([ccu1, ccu2, ccu3]))

cpb1 = c(pb1) 
cpb2 = c(pb2)
cpb3 = c(pb3)
cpbm = unp.uarray(*avg_and_sem([cpb1, cpb2, cpb3]))

cgr1 = c(gr1) 
cgr2 = c(gr2)
cgr3 = c(gr3)
cgrm = unp.uarray(*avg_and_sem([cgr1, cgr2, cgr3]))

wcu1 = [ccu1, 63.5, 16.8, 136, 8.96, 24.9]
wcu2 = [ccu2, 63.5, 16.8, 136, 8.96, 24.7]
wcu3 = [ccu3, 63.5, 16.8, 136, 8.96, 24.4]

wpb1 = [cpb1, 207.2, 29, 42, 11.35, 25.4]
wpb2 = [cpb2, 207.2, 29, 42, 11.35, 24.1]
wpb3 = [cpb3, 207.2, 29, 42, 11.35, 24.2]

wgr1 = [cgr1, 12, 8, 33, 2.25, 24.5]
wgr2 = [cgr2, 12, 8, 33, 2.25, 24.1]
wgr3 = [cgr3, 12, 8, 33, 2.25, 24.0]

CVcu1 = CV(wcu1)
CVcu2 = CV(wcu2)
CVcu3 = CV(wcu3)
CVcum = unp.uarray(*avg_and_sem([CVcu1, CVcu2, CVcu3]))

CVpb1 = CV(wpb1)
CVpb2 = CV(wpb2)
CVpb3 = CV(wpb3)
CVpbm = unp.uarray(*avg_and_sem([CVpb1, CVpb2, CVpb3]))

CVgr1 = CV(wgr1)
CVgr2 = CV(wgr2)
CVgr3 = CV(wgr3)
CVgrm = unp.uarray(*avg_and_sem([CVgr1, CVgr2, CVgr3]))

print('cgmg: ',cgmg)
print('Kupfer: ')
print('ccu: ')
print('ccu1: ',ccu1)
print('ccu2: ',ccu2)
print('ccu3: ',ccu3)
print('ccum: ',ccum)
print('CVcu: ')
print('CVcu1: ',CVcu1)
print('CVcu2: ',CVcu2)
print('CVcu3: ',CVcu3)
print('CVcum: ',CVcum)
print('Blei: ')
print('cpb: ')
print('cpb1: ',cpb1)
print('cpb2: ',cpb2)
print('cpb3: ',cpb3)
print('cpbm: ',cpbm)
print('CVpb: ')
print('CVpb1: ',CVpb1)
print('CVpb2: ',CVpb2)
print('CVpb3: ',CVpb3)
print('CVpbm: ',CVpbm)
print('Graphit: ')
print('cgr: ')
print('cgr1: ',cgr1)
print('cgr2: ',cgr2)
print('cgr3: ',cgr3)
print('cgrm: ',cgrm)
print('CVgr: ')
print('CVgr1: ',CVgr1)
print('CVgr2: ',CVgr2)
print('CVgr3: ',CVgr3)
print('CVgrm: ',CVgrm)

ersteSpalte = convert((r'$m_\text{K}/\si{\gram}$',r'$m_\text{W}/\si{\gram}$',r'$T_\text{K}/\si{\kelvin}$',r'$T_\text{W}/\si{\kelvin}$',r'$T_\text{M}/\si{\kelvin}$'),strFormat)
zweiteSpalte = convert(np.array([cu1[0],cu1[2],cu1[1]]+cu1[3:]),floatFormat,[['',r'1.2f',False],['',r'1.2f',False],['',r'1.1f',False],['',r'1.1f',False],['',r'1.1f',False]])
dritteSpalte = convert(np.array([cu2[0],cu2[2],cu2[1]]+cu2[3:]),floatFormat,[['',r'1.2f',False],['',r'1.2f',False],['',r'1.1f',False],['',r'1.1f',False],['',r'1.1f',False]])
vierteSpalte = convert(np.array([cu3[0],cu3[2],cu3[1]]+cu3[3:]),floatFormat,[['',r'1.2f',False],['',r'1.2f',False],['',r'1.1f',False],['',r'1.1f',False],['',r'1.1f',False]])
makeNewTable([ersteSpalte,zweiteSpalte, dritteSpalte, vierteSpalte],'{}&{Messung 1}&{Messung 2}&{Messung 3}','tab2-1', ['c','S[table-format=3.2]',r'S[table-format=3.2]','S[table-format=3.2]'])


ersteSpalte = convert((r'$c_\text{K}/\si[per-mode=fraction]{\joule\per\gram\per\kelvin}$',r'$C_\text{V}/\si[per-mode=fraction]{\joule\per\mol\per\kelvin}$'),strFormat)
zweiteSpalte = convert(np.array([ccu1]+[CVcu1]),floatFormat,['',r'1.2f',False])
dritteSpalte = convert(np.array([ccu1]+[CVcu2]),floatFormat,['',r'1.2f',False])
vierteSpalte = convert(np.array([ccu1]+[CVcu3]),floatFormat,['',r'1.2f',False])
makeNewTable([ersteSpalte,zweiteSpalte, dritteSpalte, vierteSpalte],'{}&{Messung 1}&{Messung 2}&{Messung 3}','tab2-2', ['c','S[table-format=3.2]',r'S[table-format=3.2]','S[table-format=3.2]'])



ersteSpalte = convert((r'$m_\text{K}/\si{\gram}$',r'$m_\text{W}/\si{\gram}$',r'$T_\text{K}/\si{\kelvin}$',r'$T_\text{W}/\si{\kelvin}$',r'$T_\text{M}/\si{\kelvin}$'),strFormat)
zweiteSpalte = convert(np.array([pb1[0],pb1[2],pb1[1]]+pb1[3:]),floatFormat,[['',r'1.2f',False],['',r'1.2f',False],['',r'1.1f',False],['',r'1.1f',False],['',r'1.1f',False]])
dritteSpalte = convert(np.array([pb2[0],pb2[2],pb2[1]]+pb2[3:]),floatFormat,[['',r'1.2f',False],['',r'1.2f',False],['',r'1.1f',False],['',r'1.1f',False],['',r'1.1f',False]])
vierteSpalte = convert(np.array([pb3[0],pb3[2],pb3[1]]+pb3[3:]),floatFormat,[['',r'1.2f',False],['',r'1.2f',False],['',r'1.1f',False],['',r'1.1f',False],['',r'1.1f',False]])
makeNewTable([ersteSpalte,zweiteSpalte, dritteSpalte, vierteSpalte],'{}&{Messung 1}&{Messung 2}&{Messung 3}','tab3-1', ['c','S[table-format=3.2]',r'S[table-format=3.2]','S[table-format=3.2]'])


ersteSpalte = convert((r'$c_\text{K}/\si[per-mode=fraction]{\joule\per\gram\per\kelvin}$',r'$C_\text{V}/\si[per-mode=fraction]{\joule\per\mol\per\kelvin}$'),strFormat)
zweiteSpalte = convert(np.array([cpb1]+[CVpb1]),floatFormat,['',r'1.2f',False])
dritteSpalte = convert(np.array([cpb1]+[CVpb2]),floatFormat,['',r'1.2f',False])
vierteSpalte = convert(np.array([cpb1]+[CVpb3]),floatFormat,['',r'1.2f',False])
makeNewTable([ersteSpalte,zweiteSpalte, dritteSpalte, vierteSpalte],'{}&{Messung 1}&{Messung 2}&{Messung 3}','tab3-2', ['c','S[table-format=3.2]',r'S[table-format=3.2]','S[table-format=3.2]'])



ersteSpalte = convert((r'$m_\text{K}/\si{\gram}$',r'$m_\text{W}/\si{\gram}$',r'$T_\text{K}/\si{\kelvin}$',r'$T_\text{W}/\si{\kelvin}$',r'$T_\text{M}/\si{\kelvin}$'),strFormat)
zweiteSpalte = convert(np.array([gr1[0],gr1[2],gr1[1]]+gr1[3:]),floatFormat,[['',r'1.2f',False],['',r'1.2f',False],['',r'1.1f',False],['',r'1.1f',False],['',r'1.1f',False]])
dritteSpalte = convert(np.array([gr2[0],gr2[2],gr2[1]]+gr2[3:]),floatFormat,[['',r'1.2f',False],['',r'1.2f',False],['',r'1.1f',False],['',r'1.1f',False],['',r'1.1f',False]])
vierteSpalte = convert(np.array([gr3[0],gr3[2],gr3[1]]+gr3[3:]),floatFormat,[['',r'1.2f',False],['',r'1.2f',False],['',r'1.1f',False],['',r'1.1f',False],['',r'1.1f',False]])
makeNewTable([ersteSpalte,zweiteSpalte, dritteSpalte, vierteSpalte],'{}&{Messung 1}&{Messung 2}&{Messung 3}','tab4-1', ['c','S[table-format=3.2]',r'S[table-format=3.2]','S[table-format=3.2]'])


ersteSpalte = convert((r'$c_\text{K}/\si[per-mode=fraction]{\joule\per\gram\per\kelvin}$',r'$C_\text{V}/\si[per-mode=fraction]{\joule\per\mol\per\kelvin}$'),strFormat)
zweiteSpalte = convert(np.array([cgr1]+[CVgr1]),floatFormat,['',r'1.2f',False])
dritteSpalte = convert(np.array([cgr1]+[CVgr2]),floatFormat,['',r'1.2f',False])
vierteSpalte = convert(np.array([cgr1]+[CVgr3]),floatFormat,['',r'1.2f',False])
makeNewTable([ersteSpalte,zweiteSpalte, dritteSpalte, vierteSpalte],'{}&{Messung 1}&{Messung 2}&{Messung 3}','tab4-2', ['c','S[table-format=3.2]',r'S[table-format=3.2]','S[table-format=3.2]'])
