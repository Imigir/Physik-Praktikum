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
import uncertainties 
import scipy.constants as const
from errorfunkt2tex import error_to_tex
from errorfunkt2tex import scipy_to_unp
from matplotlib.legend_handler import (HandlerLineCollection,HandlerTuple)
#from sympy import *
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




def gaus(x, a, c,sigma,b):
    return a* np.exp(-(x-b)**2/(2*sigma**2))+c

def Line(x, a, b):
    return a* x+b

"""
xWerte = range(0,100)
E = np.array(xWerte)+0.1
E*=0
bs=[]
sigmas=[]
for number in range(0,4):
    sigma=(1+random.random())
    sigmas.append(sigma)
    a=random.random()*3+1
    b=random.random()*xWerte[-1]+xWerte[0]
    bs.append(b)
    E2=[]
    for i in xWerte:
        E2.append((random.random()+8)/9*gaus(i,a,0,sigma,b))
    E2=np.array(E2)
    E+=E2
sigmas=np.array(sigmas)
bs=np.array(bs)
"""

def Plot(Werte, ranges, name, funktionParams=(1,0), onlyname=False, xname='$K$'):
    for rangeVar in ranges:
        plt.cla()
        plt.clf()
        #print(Werte[rangeVar[0]-1:rangeVar[1]]!=0)
        plt.plot(Line(np.array(range(rangeVar[0],rangeVar[1]+1))[Werte[rangeVar[0]-1:rangeVar[1]]!=0],*funktionParams), (Werte[rangeVar[0]-1:rangeVar[1]])[Werte[rangeVar[0]-1:rangeVar[1]]!=0], 'gx', label='Wertepaare')  
        plt.xlabel(xname)
        plt.ylabel(r'$N$')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
        if onlyname:
            plt.savefig('build/'+name+'.png') 
        else:
            plt.savefig('build/'+name+'_'+str(rangeVar[0])+'-'+ str(rangeVar[1])+'.png') 

def gausFitMitPlot(Werte, ranges, name, plotF=False, funktionParams=(1,0)):
    AllParams = []
    for rangeVar in ranges:
        p0=[np.max(Werte[rangeVar[0]-1:rangeVar[1]])-np.min(Werte[rangeVar[0]-1:rangeVar[1]]),np.min(Werte[rangeVar[0]-1:rangeVar[1]]),funktionParams[0]*((rangeVar[1]-rangeVar[0])/15),Line((rangeVar[1]+rangeVar[0])/2,*funktionParams)]
        params, covar = curve_fit(gaus,Line(np.array(range(rangeVar[0],rangeVar[1]+1)),*funktionParams),Werte[rangeVar[0]-1:rangeVar[1]],maxfev=10000,p0=p0)
        AllParams.append(uncertainties.correlated_values(params, covar))
        if plotF:
            plt.cla()
            plt.clf()
            x=np.linspace(rangeVar[0]-0.02*(rangeVar[1]-rangeVar[0]),rangeVar[1]+0.02*(rangeVar[1]-rangeVar[0]),1000)
            plt.plot(range(rangeVar[0],rangeVar[1]+1), Werte[rangeVar[0]-1:rangeVar[1]], 'gx', label='Werte')  
            plt.plot(x, gaus(x,*params), 'r-', label='Fit')
            #plt.plot(x, gaus(x,*p0), 'b-', label='Fit geschätzt')
            plt.xlim(x[0],x[-1]) 
            plt.xlabel(r'$K$')
            plt.ylabel(r'$N$')
            plt.legend(loc='best')
            plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
            plt.savefig('build/'+name+'_'+str(rangeVar[0])+'-'+ str(rangeVar[1])+'.png') 
    
    return np.array(AllParams)


#######################
#print(gausFitMitPlot(E,[[0,25]],'test'))

#for i in range(len(bs)):
#    print(bs[i])
#    print(sigmas[i])
#    print([np.max(int(bs[i]-sigmas[i]*5)-1,0),int(bs[i]+sigmas[i]*5)+1])
#    print(gausFitMitPlot(E,[[np.max(np.array([int(bs[i]-sigmas[i]*5)-1,1])),np.min(np.array([int(bs[i]+sigmas[i]*5)+1,len(E)]))]],'test'))

#print(gausFitMitPlot(E,[[0,100]],'test'))

#####################################################kali


#             0         0        121       244       295       344       367        411         443         688         778         867         964         1085        1112        1212        1299        1408         1457
ranges = [[100,115],[115,126],[300,320],[605,625],[735,750],[853,870],[910,930],[1020,1035],[1100,1118],[1700,1730],[1930,1950],[2145,2175],[2370,2425],[2680,2720],[2740,2790],[3005,3030],[3200,3260],[3460,3540],[3590,3665]]
energies= unp.uarray([0,0,121.7817,244.6974,295.9387,344.2785,367.7891,411.1165,443.9606,688.670,778.9045,867.380,964.057,1085.837,1112.076,1212.948,1299.142,1408.013,1457.643],[0,0,0.0003,0.0008,0.0017,0.0012,0.0020,0.0012,0.0016,0.005,0.0024,0.003,0.005,0.010,0.003,0.011,0.008,0.003,0.011])
wahrscheinlichkeiten = unp.uarray([0,0,107.3,28.39,1.656,100.0,3.232,8.413,10.63,3.221,48.62,15.90,54.57,38.04,51.40,5.320,6.14,78.48,1.869],[0,0,0.4,0.10,0.015,0.6,0.015,0.026,0.03,0.019,0.22,0.09,0.13,0.10,0.23,0.021,0.03,0.13,0.014])
wahrscheinlichkeiten*= unp.uarray(0.2659,0.0013)/100
EU152 = np.genfromtxt('scripts/Europium.txt',unpack=True)
print('EU152')
Plot(EU152,[[1,8192],[1,2000],[2000,4000]],'EU152')
EU152Params=gausFitMitPlot(EU152,ranges,'EU152',plotF=False)
#print(EU152Params)
pos=[]
sigma=[]
a=[]
hU=[]
for params in EU152Params:
    pos.append(params[3])
    sigma.append(params[2])
    a.append(params[0])
    hU.append(params[1])

a[8] = a[8]*0.8 #wegen Außreißer

hU=np.array(hU)
posU=np.array(pos)
pos=unp.nominal_values(pos)
posStd=unp.std_devs(pos)
sigmaU=np.array(sigma)
sigma=unp.nominal_values(sigma)
sigmaStd=unp.std_devs(sigma)
aU=np.array(a)
a=unp.nominal_values(a)
aStd=unp.std_devs(a)
wahrscheinlichkeitenN=unp.nominal_values(wahrscheinlichkeiten)
nurUeberProzent=0.02
GenommeneWerteEu=wahrscheinlichkeitenN>nurUeberProzent
nichtGenommeneWerteEu=np.logical_and(GenommeneWerteEu==False,wahrscheinlichkeitenN!=0)
x=np.linspace(1,4000,1000)
params, covar = curve_fit(Line,pos[GenommeneWerteEu], unp.nominal_values(energies[GenommeneWerteEu]))
umrechnungsParams=uncertainties.correlated_values(params, covar)
print(umrechnungsParams)
plt.cla()
plt.clf()
xA=posU[GenommeneWerteEu]
yA=energies[GenommeneWerteEu]
xA2=posU[nichtGenommeneWerteEu]
yA2=energies[nichtGenommeneWerteEu]
plt.errorbar(unp.nominal_values(xA), unp.nominal_values(yA), yerr=unp.std_devs(yA), xerr=unp.std_devs(xA), label='gefittete Wertepaare',fmt='x', capthick=0.5, linewidth='0.5',ecolor='b',capsize=1,markersize=1.5) 
plt.errorbar(unp.nominal_values(xA2), unp.nominal_values(yA2), yerr=unp.std_devs(yA2), xerr=unp.std_devs(xA2), label='nicht gefittete Wertepaare',fmt='gx', capthick=0.5, linewidth='0.5',ecolor='g',capsize=1,markersize=1.5) 
plt.plot(x, Line(x, *params), 'r-', label='Fit')
#print('x0:',x[0])
plt.xlim(0,4000) 
plt.xlabel(r'$K$')
plt.ylabel(r'$E_\gamma/\si{\kilo\electronvolt}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/EnergieKali.png') 


def polynom3(x,a,b,c,d,f):
    return f*x**4+d*x**3+a*x**2+b*x+c

def expC(x,a,b):
    return a*np.exp(-(x/1000-b))

def potenzFunktion(x, a, b, p):
    return a*(x-b)**p

def potenzFunktion(x, a, p):
    return a*x**p

a=7.31 +1.5 #cm
r=2.25 #cm
omegaDurch4PI = (1-a/np.sqrt(a**2+r**2))/2
print('omega/4pi',omegaDurch4PI)
AnzahlAnTagen=unp.uarray(6626,2) #days
HalbwertsZeit=unp.uarray(4943,5) #days
AktivitätEu=unp.uarray(4130,60) #bq
AktivitätEu=1/2**(AnzahlAnTagen/HalbwertsZeit) * AktivitätEu
print('aktivität Eu',AktivitätEu)

#print('Qinsgesammt alle einträge', np.sum(EU152)/(AktivitätEu*omegaDurch4PI*2723))
x=np.linspace(0,3700,1000)
inhalt=aU*(np.sqrt(2*np.pi)*sigmaU)
xA=posU[GenommeneWerteEu]
xA2=posU[nichtGenommeneWerteEu]
Messzeit=2723
yA=inhalt[GenommeneWerteEu]/(AktivitätEu*omegaDurch4PI*Messzeit*wahrscheinlichkeiten[GenommeneWerteEu])
yA2=inhalt[nichtGenommeneWerteEu]/(AktivitätEu*omegaDurch4PI*Messzeit*wahrscheinlichkeiten[nichtGenommeneWerteEu])
zukleinxA=Line(xA[0],*unp.nominal_values(umrechnungsParams))
xA=Line(xA[1:],*unp.nominal_values(umrechnungsParams))
xA2=Line(xA2,*unp.nominal_values(umrechnungsParams))
zukleinyA=yA[0]
yA=yA[1:]
params, covar = curve_fit(potenzFunktion, unp.nominal_values(xA), unp.nominal_values(yA),maxfev=10000,sigma=unp.std_devs(yA))
paramsEQU=uncertainties.correlated_values(params, covar)
print('Q Params', paramsEQU)
plt.cla()
plt.clf()
plt.errorbar(unp.nominal_values(xA), unp.nominal_values(yA), yerr=unp.std_devs(yA), xerr=unp.std_devs(xA), label='gefittete Wertepaare',fmt='x', capthick=0.5, linewidth='0.5',ecolor='b',capsize=1,markersize=1.5) 
plt.errorbar(unp.nominal_values(xA2), unp.nominal_values(yA2), yerr=unp.std_devs(yA2), xerr=unp.std_devs(xA2), label='nicht gefittete Wertepaare',fmt='gx', capthick=0.5, linewidth='0.5',ecolor='g',capsize=1,markersize=1.5) 
plt.errorbar(unp.nominal_values(zukleinxA), unp.nominal_values(zukleinyA), yerr=unp.std_devs(zukleinyA), xerr=unp.std_devs(zukleinxA),fmt='gx', capthick=0.5, linewidth='0.5',ecolor='g',capsize=1,markersize=1.5) 
plt.plot(x, potenzFunktion(x, *params), 'r-', label='Fit') 
plt.xlim(100,1500)
plt.ylim(0,0.5)
plt.xlabel(r'$E_\gamma/\si{\kilo\electronvolt}$')
plt.ylabel(r'$Q$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/Q.png') 

"""
range1=[]
range2=[]
for rangew in ranges:
    range1.append(rangew[0])
    range2.append(rangew[1])
range1=np.array(range1)
range2=np.array(range2)
"""

makeNewTable(convert([energies[2:]],unpFormat,[r'','4.2f',True])+convert([wahrscheinlichkeiten[2:]*100],unpFormat,[r'','2.2f',True])+convert([posU[2:]],unpFormat,[r'','3.1f',True])+convert([sigmaU[2:]],unpFormat,[r'','1.1f',True])+convert([aU[2:]],unpFormat,[r'','4.0f',True])+convert([hU[2:]],unpFormat,[r'','3.1f',True]),r'{$E_\gamma^{\text{lit,\cite{MARTIN20131497}}}/\si{\kilo\electronvolt}$} & {$W^\text{\cite{MARTIN20131497}}/\si{\percent}$} & {$b$} & {$\sigma$} & {$a$} & {$c$}','a',['S[table-format=4.2]','S[table-format=2.2]','S[table-format=3.1]','S[table-format=1.1]','S[table-format=4.1]','S[table-format=3.0]'],[r'{:1.0f}',r'{:1.0f}',r'{:1.0f}',r'{:1.0f}',r'{:1.0f}',r'{:1.0f}'])
makeNewTable(convert([energies[2:]],unpFormat,[r'','4.2f',True])+convert([Line(posU[2:],*umrechnungsParams)],unpFormat,[r'','4.2f',True])+convert([wahrscheinlichkeiten[2:]*100],unpFormat,[r'','2.2f',True])+convert([inhalt[2:]],unpFormat,[r'','5.0f',True])+convert([inhalt[2:]/(AktivitätEu*omegaDurch4PI*Messzeit*wahrscheinlichkeiten[2:])],unpFormat,[r'','0.3f',True]),r'{$E_\gamma^{\text{lit,\cite{MARTIN20131497}}}/\si{\kilo\electronvolt}$} & {$E_\gamma$} & {$W^\text{\cite{MARTIN20131497}}/\si{\percent}$} & {$Z$} & {$Q$}','a2',['S[table-format=4.2]','S[table-format=4.2]','S[table-format=2.2]','S[table-format=5.0]','S[table-format=0.3]'])

#rangeVar=[1,8192]
#xA=Line(np.array(range(rangeVar[0],rangeVar[1]+1)),*unp.nominal_values(umrechnungsParams))
#yA=EU152[rangeVar[0]-1:rangeVar[1]]
#plt.cla()
#plt.clf()
#plt.plot(xA, yA, 'gx', label='Werte') 
#plt.xlabel(r'$E_\gamma/\si{\kilo\electronvolt}$')
#plt.ylabel(r'$N$')
##plt.ylim(0,175)
##plt.xlim(0,500)
#plt.legend(loc='best')
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/EU152.pdf')
#Plot(EU152,[[1,8192]],'EU152', (1,0), True)

######################################################Cs137

Cs137 = np.genfromtxt('scripts/Cs.txt',unpack=True)
rangeVar=[1,2000]
Plot(Cs137,[rangeVar],'Cs137', unp.nominal_values(umrechnungsParams), True, r'$E_\gamma$')
#xA=Line(np.array(range(rangeVar[0],rangeVar[1]+1)),*unp.nominal_values(umrechnungsParams))
#yA=Cs137[rangeVar[0]-1:rangeVar[1]]
#plt.cla()
#plt.clf()
#plt.plot(xA, yA, 'gx', label='Werte') 
#plt.xlabel(r'$E_\gamma/\si{\kilo\electronvolt}$')
#plt.ylabel(r'$N$')
#plt.ylim(0,175)
#plt.xlim(0,500)
#plt.legend(loc='best')
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/Cs137.pdf')
ranges = [[1635,1660],[400,550]]
print('Cs137')
peakCs137=gausFitMitPlot(Cs137,ranges,'Cs137', True)

#range1=[]
#range2=[]
aU=[]
posU=[]
sigmaU=[]
hU=[]
#for rangew in ranges:
#    range1.append(rangew[0])
#    range2.append(rangew[1])
for param in peakCs137:
    aU.append(param[0])
    posU.append(param[3])
    sigmaU.append(param[2])
    hU.append(param[1])

aU=np.array(aU)
posU=np.array(posU)
sigmaU=np.array(sigmaU)
hU=np.array(hU)
#range1=np.array(range1)
#range2=np.array(range2)
makeNewTable(convert([Line(posU,*umrechnungsParams)],unpFormat,[r'','4.1f',True])+convert([posU],unpFormat,[r'','5.1f',True])+convert([sigmaU],unpFormat,[r'','3.1f',True])+convert([aU],unpFormat,[r'','4.0f',True])+convert([hU],unpFormat,[r'','2.0f',True]),r'{$E_\gamma/\si{\kilo\electronvolt}$} & {$b$} & {$\sigma$} & {$a$} & {$c$}','b',['S[table-format=3.1]','S[table-format=4.1]','S[table-format=2.1]','S[table-format=4.0]','S[table-format=2.0]'],[r'{:1.0f}',r'{:1.0f}',r'{:1.0f}',r'{:1.0f}',r'{:1.0f}'])

#print('PeakParams', peakCs137)
A0=(range(1635,1660+1),Cs137[1635-1:1660])
A1=(range(1643,1645+1),Cs137[1643-1:1645])
A2=(range(1645,1648+1),Cs137[1645-1:1648])
A3=(range(1650,1653+1),Cs137[1650-1:1653])
A4=(range(1653,1655+1),Cs137[1653-1:1655])
print('photoPeak')
params, covar = curve_fit(Line, *A1,maxfev=10000)
params1=uncertainties.correlated_values(params, covar)
params, covar = curve_fit(Line, *A2,maxfev=10000)
params2=uncertainties.correlated_values(params, covar)
params, covar = curve_fit(Line,*A3,maxfev=10000)
params3=uncertainties.correlated_values(params, covar)
params, covar = curve_fit(Line, *A4,maxfev=10000)
params4=uncertainties.correlated_values(params, covar)

print('peakhöhe',np.max(Cs137[1635-1:1660]))
halbeHöhe=np.max(Cs137[1635-1:1660])/2
zehntelHöhe=np.max(Cs137[1635-1:1660])/10
zehntelBreite=-(zehntelHöhe-params1[1])/params1[0]+(zehntelHöhe-params4[1])/params4[0]
halbeBreite=-(halbeHöhe-params2[1])/params2[0]+(halbeHöhe-params3[1])/params3[0]
print('zehntelBreite', zehntelBreite)
print('halbeBreite', halbeBreite)
print('zehntelBreiteUmgerechnet', zehntelBreite*umrechnungsParams[0])
print('halbeBreiteUmgerechnet', halbeBreite*umrechnungsParams[0])
print('jk', zehntelBreite/halbeBreite)
E_El=2.9*10**(-3) #keV
print('halbeBreiteBerrechnet', unp.sqrt(np.log(2)*8 *0.1 * Line(peakCs137[0][3],*umrechnungsParams)* E_El))

x=np.linspace(1635,1660)
plt.cla()
plt.clf()
mm1, = plt.plot(Line(A0[0],*unp.nominal_values(umrechnungsParams)),A0[1], 'gx', label='Werte0')  
mm2, = plt.plot(Line(A1[0],*unp.nominal_values(umrechnungsParams)),A1[1], 'bx', label='Werte1')  
mm3, = plt.plot(Line(A4[0],*unp.nominal_values(umrechnungsParams)),A4[1], 'yx', label='Werte4') 
mm4, = plt.plot(Line(x,*unp.nominal_values(umrechnungsParams)), x*0+zehntelHöhe, 'r-', label='Fit')
mm5, = plt.plot(Line(x,*unp.nominal_values(umrechnungsParams)), Line(x,*unp.nominal_values(params1)), 'b-', label='Fit')
mm6, = plt.plot(Line(x,*unp.nominal_values(umrechnungsParams)), Line(x,*unp.nominal_values(params4)), 'y-', label='Fit')
plt.ylim(-10,max(Cs137[1635-1:1660])/4)
#plt.plot(x, gaus(x,*p0), 'b-', label='Fit geschätzt')
plt.xlabel(r'$E_\gamma$')
plt.ylabel(r'$N$')
plt.legend([(mm1, mm2, mm3), mm4, mm5, mm6], ['Wertepaare','Zehntel der Höhe','Fit der linken Flanke','Fit der rechten Flanke'],handler_map={tuple: HandlerTuple(ndivide=None)},loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/Cs137Zehntel.png') 

plt.cla()
plt.clf()
mm1, = plt.plot(Line(A0[0],*unp.nominal_values(umrechnungsParams)),A0[1], 'gx', label='Werte0') 
mm2, = plt.plot(Line(A2[0],*unp.nominal_values(umrechnungsParams)),A2[1], 'yx', label='Werte2') 
mm3, = plt.plot(Line(A3[0],*unp.nominal_values(umrechnungsParams)),A3[1], 'rx', label='Werte3') 
mm4, = plt.plot(Line(x,*unp.nominal_values(umrechnungsParams)), x*0+halbeHöhe, 'b-', label='Fit')
mm5, = plt.plot(Line(x,*unp.nominal_values(umrechnungsParams)), Line(x,*unp.nominal_values(params2)), 'y-', label='Fit')
mm6, = plt.plot(Line(x,*unp.nominal_values(umrechnungsParams)), Line(x,*unp.nominal_values(params3)), 'r-', label='Fit')
plt.ylim(-10,max(Cs137[1635-1:1660])+100)
plt.xlabel(r'$E_\gamma$')
plt.ylabel(r'$N$')
plt.legend([(mm1, mm2, mm3), mm4, mm5, mm6], ['Wertepaare','Hälfte der Höhe','Fit der linken Flanke','Fit der rechten Flanke'],handler_map={tuple: HandlerTuple(ndivide=None)},loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/Cs137Halb.png') 

makeNewTable(convert([np.array([params1[0],params2[0],params3[0],params4[0]])],unpFormat,[r'','2.0f',True])+convert([np.array([params1[1],params2[1],params3[1],params4[1]])],unpFormat,[r'','2.0f',True]),r'{$a$} & {$b$}','geraden1',['S[table-format=4.2]','S[table-format=4.2]'])

##########################################################add

def diffWirkung(E,c,h):
    Egamma=Line(unp.nominal_values(peakCs137[0][3]),*unp.nominal_values(umrechnungsParams))
    r=const.elementary_charge/(4*np.pi * const.epsilon_0 * const.electron_mass*const.c**2)
    m0=const.electron_mass*const.c**2 / (1000*const.electron_volt)
    e=Egamma/m0
    th=8/3 * np.pi * r**2
    return 3/8 * th* 1/(m0*e**2) * (2+(E/(Egamma-E))**2 * (1/e**2 + (1-2/e)* (Egamma-E)/Egamma )) * c+h

def Wirkungin(c,h):
    Egamma=Line(unp.nominal_values(peakCs137[0][3]),*unp.nominal_values(umrechnungsParams))
    r=const.elementary_charge/(4*np.pi * const.epsilon_0 * const.electron_mass*const.c**2)
    m0=const.electron_mass*const.c**2 / (1000*const.electron_volt)
    e=Egamma/m0
    th=8/3 * np.pi * r**2
    term1 = 3*Egamma*c*th
    term2 = 2*e*(2+e*(8+e*(11+e)))
    term3 = (1+2*e)**2 * ((e-2)*e-2)*np.log(1+2*e)
    term4 = 8*e**4* (1+2*e)**2 *m0
    term5 = (2*e*Egamma*h)/(1+2*e)
    return term1*(term2+term3)/term4 + term5


print('ComptonKontinuum')

rangeVar=[800,1170]
xA=Line(np.array(range(rangeVar[0],rangeVar[1]+1)),*unp.nominal_values(umrechnungsParams))
yA=Cs137[rangeVar[0]-1:rangeVar[1]]
params, covar = curve_fit(diffWirkung, xA, yA)
paramsKon=uncertainties.correlated_values(params, covar)
#print('paramsKon',paramsKon)
rangeVar=[1,1250]
xA2=Line(np.array(range(rangeVar[0],rangeVar[1]+1)),*unp.nominal_values(umrechnungsParams))
yA2=Cs137[rangeVar[0]-1:rangeVar[1]]
x=np.linspace(rangeVar[0],rangeVar[1],1000)
x=Line(x,*unp.nominal_values(umrechnungsParams))
plt.cla()
plt.clf()
plt.plot(x, diffWirkung(x, *params), 'r-', label='Fit des Compton-Kontinuums') 
plt.plot(xA2, yA2, 'gx', label='Wertepaare') 
plt.plot(xA, yA, 'bx', label='In den Fit mit einbezogene Wertepaare')
Egamma=Line(unp.nominal_values(peakCs137[0][3]),*umrechnungsParams)
print('EgammaPhoto', Egamma)
print('ERück', Line(unp.nominal_values(peakCs137[1][3]),*umrechnungsParams))
I_Compton = (Wirkungin(*paramsKon)/unp.nominal_values(umrechnungsParams[0]))
print('IntegralCompton', I_Compton)
I_Ph = (peakCs137[0][0]*(np.sqrt(2*np.pi)*peakCs137[0][2]))
print('IntegralPhotoPeak', I_Ph)
print('IntegralVerhälnisExperiment', I_Compton/I_Ph)
print('IntegralVerhältnisTheorie', (1-unp.exp(-3.9*unp.uarray(3.7,0.1)))/(1-unp.exp(-3.9*unp.uarray(0.0075,0.0005))))

m0=const.electron_mass*const.c**2 / (1000*const.electron_volt)
e=Egamma/m0
Emax=unp.nominal_values(Egamma * 2*e/(1+2*e))
print('EmaxBer', Egamma * 2*e/(1+2*e))
print('ErückBer', Egamma-Emax)
plt.plot(np.array([Emax,Emax]), np.array([0,180]), 'y-', label='berechnete Lage der Compton-Kante')  
plt.xlabel(r'$E_\gamma/\si{\kilo\electronvolt}$')
plt.ylabel(r'$N$')
plt.ylim(0,175)
plt.xlim(0,500)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/Cs137Kon.png')

rangeVar=[1000,1250]
xA=Line(np.array(range(rangeVar[0],rangeVar[1]+1)),*unp.nominal_values(umrechnungsParams))
yA=Cs137[rangeVar[0]-1:rangeVar[1]]
rangeVar2=[1000,1170]
xA1=Line(np.array(range(rangeVar2[0],rangeVar2[1]+1)),*unp.nominal_values(umrechnungsParams))
yA1=Cs137[rangeVar2[0]-1:rangeVar2[1]]
rangeVar3=[1170,1210]
xA2=Line(np.array(range(rangeVar3[0],rangeVar3[1]+1)),*unp.nominal_values(umrechnungsParams))
yA2=Cs137[rangeVar3[0]-1:rangeVar3[1]]
params, covar = curve_fit(Line, xA1, yA1)
paramsEmax1=uncertainties.correlated_values(params, covar)
params, covar = curve_fit(Line, xA2, yA2)
paramsEmax2=uncertainties.correlated_values(params, covar)

x=np.linspace(rangeVar[0]-10,rangeVar[1]+10,1000)
x=Line(x,*unp.nominal_values(umrechnungsParams))
plt.cla()
plt.clf()
mm1, = plt.plot(xA, yA, 'gx', label='Werte') 
mm2, = plt.plot(xA1, yA1, 'bx', label='Werte') 
mm3, = plt.plot(xA2, yA2, 'rx', label='Werte') 
mm4, = plt.plot(x, Line(x,*unp.nominal_values(paramsEmax1)), 'b-', label='Werte') 
mm5, = plt.plot(x, Line(x,*unp.nominal_values(paramsEmax2)), 'r-', label='Werte') 
print('SchnittEmax', (paramsEmax1[1]-paramsEmax2[1])/(paramsEmax2[0]-paramsEmax1[0]))
plt.xlabel(r'$E_\gamma/\si{\kilo\electronvolt}$')
plt.ylabel(r'$N$')
plt.ylim(0,100)
plt.xlim(400,500)
plt.legend([(mm1, mm2, mm3), mm4, mm5], ['Wertepaare','Fit der linken Flanke','Fit der rechten Flanke'],handler_map={tuple: HandlerTuple(ndivide=None)},loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/Cs137Emax.png')
makeNewTable(convert([np.array([paramsEmax1[0],paramsEmax2[0]])],unpFormat,[r'','2.2f',True])+convert([np.array([paramsEmax1[1],paramsEmax2[1]])],unpFormat,[r'','2.0f',True]),r'{$a$} & {$b$}','geraden2',['S[table-format=4.2]','S[table-format=4.2]'])











########################################################Ba
print('Ba')
ranges = [[1,8192]]
D = np.genfromtxt('scripts/Sb_Ba.txt',unpack=True)
#Plot(D,ranges,'D')
ranges = [[200,215],[650,740],[750,770],[880,900],[950,970]]
DParams=gausFitMitPlot(D,ranges,'D',plotF=True)
#range1=[]
#range2=[]
aU=[]
posU=[]
sigmaU=[]
hU=[]
#for rangew in ranges:
#    range1.append(rangew[0])
#    range2.append(rangew[1])
for param in DParams:
    aU.append(param[0])
    posU.append(param[3])
    sigmaU.append(param[2])
    hU.append(param[1])

aU=np.array(aU)
posU=np.array(posU)
sigmaU=np.array(sigmaU)
hU=np.array(hU)
#range1=np.array(range1)
#range2=np.array(range2)
makeNewTable(convert([Line(posU,*umrechnungsParams)],unpFormat,[r'','1.1f',True])+convert([posU],unpFormat,[r'','1.2f',True])+convert([sigmaU],unpFormat,[r'','1.2f',True])+convert([aU],unpFormat,[r'','1.0f',True])+convert([hU],unpFormat,[r'','1.1f',True]),r'{$E_\gamma/\si{\kilo\electronvolt}$} & {$b$} & {$\sigma$} & {$a$} & {$c$}','D',['S[table-format=2.0]','S[table-format=2.0]','S[table-format=2.0]','S[table-format=2.0]','S[table-format=2.0]'],[r'{:1.0f}',r'{:1.0f}',r'{:1.0f}',r'{:1.0f}',r'{:1.0f}'])

print(DParams)
print('E1', Line(DParams[0][3],*umrechnungsParams))
print('E2', Line(DParams[1][3],*umrechnungsParams))
print('E3', Line(DParams[2][3],*umrechnungsParams))
print('E4', Line(DParams[3][3],*umrechnungsParams))
print('E5', Line(DParams[4][3],*umrechnungsParams))
#Pos=[]
#Sigma=[]
#a=[]
#for param in DParams:
#    Pos.append(param[3])
#    Sigma.append(param[2])
#    a.append(param[0])
#Pos=np.array(Pos)
#Sigma=np.array(Sigma)
#a=np.array(a)
inhalt=aU*(np.sqrt(2*np.pi)*sigmaU)
t=3669
wahrscheinlichkeitenBa=unp.uarray([53.1,11.54,29.55,100,14.41],[0.5,0.07,0.18,0,0.09])*unp.uarray(0.6205,0.0019)/100
EnergieBaLit=unp.uarray([80.9979,276.3989,302.8508,356.0129,383.8485],[0.0011,0.0012,0.0005,0.0007,0.0012])
#print(EnergieBaLit)
#print(wahrscheinlichkeitenBa)
#wahrscheinlichkeitenBa=np.array([34.1,18.3,62.1,8.9])/100
A=inhalt/(potenzFunktion(Line(posU,*umrechnungsParams),*paramsEQU)*wahrscheinlichkeitenBa*omegaDurch4PI*t)
print('A',A)
print('A', *weighted_avg_and_sem(unp.nominal_values(A[1:]), 3* (1/unp.std_devs(A[1:]))/np.sum(1/unp.std_devs(A[1:]))))
rangeVar=[1,1200]
Plot(D,[rangeVar],'D', unp.nominal_values(umrechnungsParams), True, r'$E_\gamma$')
#xA=Line(np.array(range(rangeVar[0],rangeVar[1]+1)),*unp.nominal_values(umrechnungsParams))
#yA=D[rangeVar[0]-1:rangeVar[1]]
#plt.cla()
#plt.clf()
#plt.plot(xA, yA, 'gx', label='Werte') 
#plt.xlabel(r'$E_\gamma/\si{\kilo\electronvolt}$')
#plt.ylabel(r'$N$')
#plt.ylim(0,175)
#plt.xlim(0,500)
#plt.legend(loc='best')
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/D.pdf')
#makeNewTable(convert([Line(posU,*umrechnungsParams)],unpFormat,[r'','1.1f',True])+convert([Pos],unpFormat,[r'','1.2f',True])+convert([sigmaU],unpFormat,[r'','1.2f',True])+convert([aU],unpFormat,[r'','1.0f',True])+convert([hU],unpFormat,[r'','1.1f',True]),r'{$E_\gamma/\si{\kilo\electronvolt}$} & {$b$} & {$\sigma$} & {$a$} & {$c$}','D',['S[table-format=2.0]','S[table-format=2.0]','S[table-format=2.0]','S[table-format=2.0]','S[table-format=2.0]'],[r'{:1.0f}',r'{:1.0f}',r'{:1.0f}',r'{:1.0f}',r'{:1.0f}'])
makeNewTable(convert([EnergieBaLit],unpFormat,[r'','4.2f',True])+convert([Line(posU,*umrechnungsParams)],unpFormat,[r'','4.2f',True])+convert([wahrscheinlichkeitenBa*100],unpFormat,[r'','2.2f',True])+convert([inhalt],unpFormat,[r'','5.0f',True])+convert([potenzFunktion(Line(posU,*umrechnungsParams),*paramsEQU)],unpFormat,[r'','0.3f',True])+convert([A],unpFormat,[r'','4.0f',True]),r'{$E_\gamma^{\text{lit,\cite{KHAZOV2011855}}}/\si{\kilo\electronvolt}$} & {$E_\gamma$} & {$W^\text{\cite{KHAZOV2011855}}/\si{\percent}$} & {$Z$} & {$Q$} & {$A$}','D2',['S[table-format=4.2]','S[table-format=4.2]','S[table-format=2.2]','S[table-format=5.0]','S[table-format=0.3]','S[table-format=4.0]'])

"""
########################################################?c060?

print('c060')
ranges = [[1,8192]]
D = np.genfromtxt('scripts/unbekannt.txt',unpack=True)
wahrscheinlichkeitenCo=unp.uarray([99.85, 99.9826],[0.03, 0.0006])/100
EnergieCoLit=unp.uarray([1173.228,1332.492],[0.003,0.004])
#Plot(D,ranges,'unbekannt')
ranges = [[2900,2930],[3280,3340]]
DParams=gausFitMitPlot(D,ranges,'unbekannt')
#range1=[]
#range2=[]
aU=[]
posU=[]
sigmaU=[]
hU=[]
#for rangew in ranges:
#    range1.append(rangew[0])
#    range2.append(rangew[1])
for param in DParams:
    aU.append(param[0])
    posU.append(param[3])
    sigmaU.append(param[2])
    hU.append(param[1])

aU=np.array(aU)
posU=np.array(posU)
sigmaU=np.array(sigmaU)
hU=np.array(hU)
#range1=np.array(range1)
#range2=np.array(range2)
inhalt=aU*(np.sqrt(2*np.pi)*sigmaU)
t=3916
A=inhalt/(potenzFunktion(Line(posU,*umrechnungsParams),*paramsEQU)*wahrscheinlichkeitenCo*omegaDurch4PI*t)
print('A',A)
print('A', *weighted_avg_and_sem(unp.nominal_values(A), 2* (1/unp.std_devs(A))/np.sum(1/unp.std_devs(A))))

makeNewTable(convert([Line(posU,*umrechnungsParams)],unpFormat,[r'','1.1f',True])+convert([posU],unpFormat,[r'','1.2f',True])+convert([sigmaU],unpFormat,[r'','1.2f',True])+convert([aU],unpFormat,[r'','1.0f',True])+convert([hU],unpFormat,[r'','1.0f',True]),r'{$E_\gamma/\si{\kilo\electronvolt}$} & {$b$} & {$\sigma$} & {$a$} & {$c$}','unbekannt',['S[table-format=2.0]','S[table-format=2.0]','S[table-format=2.0]','S[table-format=2.0]','S[table-format=2.0]'],[r'{:1.0f}',r'{:1.0f}',r'{:1.0f}',r'{:1.0f}',r'{:1.0f}'])
makeNewTable(convert([EnergieCoLit],unpFormat,[r'','4.2f',True])+convert([Line(posU,*umrechnungsParams)],unpFormat,[r'','4.2f',True])+convert([wahrscheinlichkeitenCo*100],unpFormat,[r'','2.2f',True])+convert([inhalt],unpFormat,[r'','5.0f',True])+convert([potenzFunktion(Line(posU,*umrechnungsParams),*paramsEQU)],unpFormat,[r'','0.3f',True])+convert([A],unpFormat,[r'','4.0f',True]),r'{$E_\gamma^{\text{lit,\cite{BROWNE20131849}}}/\si{\kilo\electronvolt}$} & {$E_\gamma$} & {$W^\text{\cite{BROWNE20131849}}/\si{\percent}$} & {$Z$} & {$Q$} & {$A$}','unbekannt2',['S[table-format=4.2]','S[table-format=4.2]','S[table-format=2.2]','S[table-format=5.0]','S[table-format=0.3]','S[table-format=4.0]'])
print(DParams)
print('E1', Line(DParams[0][3],*umrechnungsParams))
print('E2', Line(DParams[1][3],*umrechnungsParams))
rangeVar=[1,4000]
Plot(D,[rangeVar],'unbekannt', unp.nominal_values(umrechnungsParams), True, r'$E_\gamma$')
#xA=Line(np.array(range(rangeVar[0],rangeVar[1]+1)),*unp.nominal_values(umrechnungsParams))
#yA=D[rangeVar[0]-1:rangeVar[1]]
#plt.cla()
#plt.clf()
#plt.plot(xA, yA, 'gx', label='Werte') 
#plt.xlabel(r'$E_\gamma/\si{\kilo\electronvolt}$')
#plt.ylabel(r'$N$')
#plt.ylim(0,175)
#plt.xlim(0,500)
#plt.legend(loc='best')
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/unbekannt.pdf')
#Pos=[]
#Sigma=[]
#a=[]
#for param in DParams:
#    Pos.append(param[3])
#    Sigma.append(param[2])
#    a.append(param[0])
#Pos=np.array(Pos)
#Pos=Pos
#Sigma=np.array(Sigma)
#Sigma=Sigma
#a=np.array(a)
#a=a
#inhalt=a*(np.sqrt(2*np.pi)*Sigma)
#t=3916
#wahrscheinlichkeitenBa=np.array([18.3,62.1,8.9])/100
#A=inhalt/(potenzFunktion(Line(Pos,*umrechnungsParams),*paramsEQU)*wahrscheinlichkeitenBa*omegaDurch4PI*t)
#print('A',A)
#print('A', *weighted_avg_and_sem(unp.nominal_values(A), 1/unp.std_devs(A) * 3/np.sum(1/unp.std_devs(A))))
"""