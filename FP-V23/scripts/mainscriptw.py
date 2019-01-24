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
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import uncertainties
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


#4.1
print(' ')
print('4.1')

def Line(x, a, b):
	return a*x+b

def Plot(x, y, limx, xname, yname, params, name, linear=True, xscale=1, yscale=1, limy=None):
	xplot = np.linspace(limx[0]*xscale,limx[1]*xscale,1000)
	plt.cla()
	plt.clf()
	plt.errorbar(noms(x)*xscale, noms(y)*yscale, xerr=stds(x)*xscale, yerr=stds(y)*yscale, fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Wertepaare')
	if(linear == True):
		plt.plot(xplot*xscale, Line(xplot, *params)*yscale, 'b-', label='Ausgleichsgerade')
	plt.xlim(limx[0]*xscale,limx[1]*xscale)
	if(limy != None):
		plt.ylim(limy[0]*yscale,limy[1]*yscale)
	plt.xlabel(xname)
	plt.ylabel(yname)
	plt.legend(loc='best')
	plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
	plt.savefig('build/'+name+'.pdf')

def PlotBandStructure(x, y, name, type='extended', a=1, numBands=1,  xscale=1, yscale=1/1000, limy=None, marker=['rx'], ranges = [0]):
	plt.cla()
	plt.clf()
	if(type == 'extended'):
		plt.errorbar(x*xscale, noms(y)*yscale, yerr=stds(y)*yscale, fmt=marker[0], markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Wertepaare')
		xlim = [0,x[-1]+10]
	if(type == 'reduced'):
		x2 = []
		marker2 = marker[0]
		for i in range(1,numBands+1):
			if(i%2 != 0):
				for k in x[(x<=i*np.pi/a) & (x>(i-1)*np.pi/a)]:
					k = k-(i-1)*np.pi/a
					x2 = np.append(x2,k)
			if(i%2 == 0):
				for k in x[(x<=i*np.pi/a) & (x>(i-1)*np.pi/a)]:
					k = -k+i*np.pi/a
					x2 = np.append(x2,k)
		if(len(ranges) != 1):
			r = ranges.copy()
			for i in range(len(ranges)):
				temp = 0
				j = 0
				while(j<=i):
					temp = temp+ranges[j]
					j += 1
				r[i] = temp
			for i in range(len(r)-1):
				if(len(marker) != 1):
					marker2 = marker[i]
				plt.errorbar(x2[r[i]:r[i+1]]*xscale, noms(y[r[i]:r[i+1]])*yscale, yerr=stds(y[r[i]:r[i+1]])*yscale, fmt=marker2, markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g', barsabove=True)
				plt.errorbar(-x2[r[i]:r[i+1]]*xscale, noms(y[r[i]:r[i+1]])*yscale, yerr=stds(y[r[i]:r[i+1]])*yscale, fmt=marker2, markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g', barsabove=True)
		x2 = np.append(x2,-x2)
		y2 = np.append(y,y)
		plt.errorbar(x2*xscale, noms(y2)*yscale, yerr=stds(y2)*yscale, fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g', barsabove=True, label='Wertepaare')
		xlim = [-np.pi/a,np.pi/a]
	plt.xlim(xlim[0]*xscale,xlim[1]*xscale)
	if(limy != None):
		plt.ylim(limy[0]*yscale,limy[1]*yscale)
	plt.xlabel(r'$k/\si{\per\metre}$')
	plt.ylabel(r'$\omega/\si{\kilo\hertz}$')
	plt.legend(loc='best')
	plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
	plt.savefig('build/'+name+'_'+type+'.pdf')

"""
def PlotBandStructure(x, y, name, type='extended', a=1, numBands=1,  xscale=1, yscale=1/1000, limy=None, marker=['rx'], ranges = [0]):
	plt.cla()
	plt.clf()
	if(type == 'extended'):
		plt.errorbar(x*xscale, noms(y)*yscale, yerr=stds(y)*yscale, fmt=marker[0], markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True ,label='Wertepaare')
		xlim = [0,x[-1]+10]
	if(type == 'reduced'):
		#x2 = []
		#y2 = []
		x3 = []
		#y3 = []
		#j = 0
		marker2 = marker[0]
		for i in range(1,numBands+1):
			#if(len(marker) != 1):
			#	marker2 = marker[i-1]
			if(i%2 != 0):
				#x2 = []
				#y2 = []
				for k in x[(x<=i*np.pi/a) & (x>(i-1)*np.pi/a)]:
					k = k-(i-1)*np.pi/a
					#x2 = np.append(x2,k)
					#y2 = np.append(y2,y[j])
					#j += 1
					x3 = np.append(x3,k)
				#temp = []
				#temp2 = []
				#for k in range(len(x2)):
				#	temp = np.append(temp,-x2[len(x2)-k-1])
				#	temp2 = np.append(temp2,y2[len(x2)-k-1])
				#x2 = np.append(temp,x2)
				#y2 = np.append(temp2,y2)
				#x2 = np.append(-x2,x2)
				#y2 = np.append(y2,y2)
				#plt.errorbar(x2*xscale, noms(y2)*yscale, yerr=stds(y2)*yscale, fmt=marker2, markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True)
			if(i%2 == 0):
				#x2 = []
				#y2 = []
				for k in x[(x<=i*np.pi/a) & (x>(i-1)*np.pi/a)]:
					k = -k+i*np.pi/a
					#x2 = np.append(x2,k)
					#y2 = np.append(y2,y[j])
					#j += 1
					x3 = np.append(x3,k)
				#temp = []
				#temp2 = []
				#print(x2)
				#for k in range(len(x2)):
				#	temp = np.append(temp,x2[len(x2)-k-1])
				#	temp2 = np.append(temp2,y2[len(x2)-k-1])
				#x2 = np.append(-x2,temp)
				#y2 = np.append(y2,temp2)
				#x2 = np.append(-x2,x2)
				#y2 = np.append(y2,y2)
				#plt.errorbar(x2*xscale, noms(y2)*yscale, yerr=stds(y2)*yscale, fmt=marker2, markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g',barsabove=True)
		if(len(ranges) != 1):
			r = ranges.copy()
			for i in range(len(ranges)):
				temp = 0
				j = 0
				while(j<=i):
					temp = temp+ranges[j]
					j += 1
				r[i] = temp
			for i in range(len(r)-1):
				if(len(marker) != 1):
					marker2 = marker[i]
				print(r[i],r[i+1]-1)
				plt.errorbar(x3[r[i]:r[i+1]]*xscale, noms(y[r[i]:r[i+1]])*yscale, yerr=stds(y[r[i]:r[i+1]])*yscale, fmt=marker2, markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g', barsabove=True)
				plt.errorbar(-x3[r[i]:r[i+1]]*xscale, noms(y[r[i]:r[i+1]])*yscale, yerr=stds(y[r[i]:r[i+1]])*yscale, fmt=marker2, markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g', barsabove=True)
		x2 = np.append(x3,-x3)
		y2 = np.append(y,y)
		plt.errorbar(x2*xscale, noms(y2)*yscale, yerr=stds(y2)*yscale, fmt='rx', markersize=6, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='g', barsabove=True, label='Wertepaare')
		xlim = [-np.pi/a,np.pi/a]
	plt.xlim(xlim[0]*xscale,xlim[1]*xscale)
	if(limy != None):
		plt.ylim(limy[0]*yscale,limy[1]*yscale)
	plt.xlabel(r'$k/\si{\per\metre}$')
	plt.ylabel(r'$\omega/\si{\kilo\hertz}$')
	plt.legend(loc='best')
	plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
	plt.savefig('build/'+name+'_'+type+'.pdf')
"""

l = np.array([75,150,225,300,375,450,525,600]) #mm
peaks = [[],[],[],[],[],[],[],[]]
f = [[],[],[],[],[],[],[],[]]
A = [[],[],[],[],[],[],[],[]]
f[0],A[0] = np.genfromtxt('FP-V23data/4.1_75mm.dat',unpack=True)
f[1],A[1] = np.genfromtxt('FP-V23data/1.2(4.1)_150mm.dat',unpack=True)
f[2],A[2] = np.genfromtxt('FP-V23data/4.1_225mm.dat',unpack=True)
f[3],A[3] = np.genfromtxt('FP-V23data/4.1_300mm.dat',unpack=True)
f[4],A[4] = np.genfromtxt('FP-V23data/4.1_375mm.dat',unpack=True)
f[5],A[5] = np.genfromtxt('FP-V23data/4.1_450mm.dat',unpack=True)
f[6],A[6] = np.genfromtxt('FP-V23data/4.1_525mm.dat',unpack=True)
f[7],A[7] = np.genfromtxt('FP-V23data/4.1_600mm.dat',unpack=True)

for i in range(len(f)):
	indices = find_peaks(A[i],12)
	peaks[i] = f[i][indices[0]]
	#print('peaks', l[i],'mm:', peaks[i])

Df_m = []
Df_err = []
for i in range(len(f)):
	Df = []
	for j in range(len(peaks[i])-1):
		temp = peaks[i][j+1]-peaks[i][j]
		Df = Df+[temp]
	Df = avg_and_sem(Df)
	Df_m = Df_m+[Df[0]]
	Df_err = Df_err+[Df[1]]
Df = unp.uarray(Df_m,Df_err)

for i in range(len(f)):
	print('Df', l[i],'mm:', Df[i],'Hz')

params, covar = curve_fit(Line,1/l*1000, noms(Df))
uParams=uncertainties.correlated_values(params, covar)

Plot(1/l*1000,Df,[1/700*1000,1/70*1000],r'$\left(\frac{1}{L}\right)/\si{\per\metre}$',r'$\Delta f/\si{\hertz}$',params,'4.1')

c = uParams[0]*2
print('m:', uParams[0],'Hz*m')
print('n:', uParams[1],'Hz')
print('c:', c,'m/s')


#4.2
print(' ')
print('4.2')

f,A = np.genfromtxt('FP-V23data/4.2_600mm.dat',unpack=True)
indices = find_peaks(A,prominence=3)
peaks = f[indices[0]]
#print('peaks:', peaks)
peaks = unp.uarray(peaks,10)*2*np.pi

n = np.linspace(2,41,40)
k = (n*np.pi)/0.6
#print('k:', k)

params, covar = curve_fit(Line,k,noms(peaks),sigma=stds(peaks),absolute_sigma=True)
uParams=uncertainties.correlated_values(params, covar)

Plot(k,peaks,[0,230],r'$k/\si{\per\metre}$',r'$\omega/\si{\kilo\hertz}$',params,'4.2',yscale=1/1000)

c = uParams[0]
print('m:', uParams[0],'Hz*m')
print('n:', uParams[1],'Hz')
print('c:', c,'m/s')



#4.3
print(' ')
print('4.3')

print('16mm')
f,A = np.genfromtxt('FP-V23data/4.3_400mm_16mm.dat',unpack=True)
indices = find_peaks(A,prominence=0.1)
peaks = f[indices[0]]
peaks = peaks[1:]
#print('peaks:', peaks)
peaks = unp.uarray(peaks,10)*2*np.pi

n = np.linspace(2,31,30)
k = (n*np.pi)/0.4
#print('k:', k)

numSegments = 8
L = 0.4
a = L/numSegments
PlotBandStructure(k,peaks,'4.3_16mm',type='extended')
PlotBandStructure(k,peaks,'4.3_16mm',type='reduced',a=a,numBands=4)

Band16 = unp.uarray([0,0,0,0],[0,0,0,0])
Band16[0] = peaks[6]-peaks[0]
Band16[1] = peaks[14]-peaks[7]
Band16[2] = peaks[22]-peaks[15]
Band16[3] = peaks[29]-peaks[23]
for i in range(0,4):
	print('Band16',(i+1),': ', Band16[i]/1000,'(/2pi)kHz')

Gap16 = unp.uarray([0,0,0],[0,0,0])
Gap16[0] = peaks[7]-peaks[6]
Gap16[1] = peaks[15]-peaks[14]
Gap16[2] = peaks[23]-peaks[22]
for i in range(0,3):
	print('Gap16',(i+1),': ', Gap16[i]/1000,'(/2pi)kHz')

#DOS
peaks = peaks/(2*np.pi)
rho_m = []
rho_err = []
for i in range(len(peaks)-1):
	rho_m = np.append(rho_m,1/(noms(peaks[i+1])-noms(peaks[i])))
	rho_err = np.append(rho_err,np.sqrt((stds(peaks[i+1])/(noms(peaks[i+1])-noms(peaks[i]))**2)**2+(stds(peaks[i])/(noms(peaks[i+1])-noms(peaks[i]))**2)**2))
rho = unp.uarray(rho_m,rho_err)

Plot(peaks[1:],rho,[100,12000],r'$\omega/\si{\kilo\hertz}$',r'$\rho/\si{\second}$',params,'4._DOS',linear=False,xscale=1/1000)


print('13mm')
f,A = np.genfromtxt('FP-V23data/4.3_400mm_13mm.dat',unpack=True)
indices = find_peaks(A,prominence=1)
peaks = f[indices[0][:7]]
indices = find_peaks(A,prominence=0.5)
peaks = np.append(peaks,f[indices[0][8:]])
#print('peaks:', peaks)
peaks = unp.uarray(peaks,10)*2*np.pi

n = np.linspace(2,16,15)
n = np.append(n,np.linspace(18,23,6))
n = np.append(n,np.linspace(25,31,6))
k = (n*np.pi)/0.4
#print('k:', k)

numSegments = 8
L = 0.4
a = L/numSegments
PlotBandStructure(k,peaks,'4.3_13mm',type='extended')
PlotBandStructure(k,peaks,'4.3_13mm',type='reduced',a=a,numBands=4)

Band13 = unp.uarray([0,0,0,0],[0,0,0,0])
Band13[0] = peaks[6]-peaks[0]
Band13[1] = peaks[14]-peaks[7]
Band13[2] = peaks[20]-peaks[15]
Band13[3] = peaks[26]-peaks[21]
for i in range(0,4):
	print('Band13',(i+1),': ', Band13[i]/1000,'(/2pi)kHz')

Gap13 = unp.uarray([0,0,0],[0,0,0])
Gap13[0] = peaks[7]-peaks[6]
Gap13[1] = peaks[15]-peaks[14]
Gap13[2] = peaks[21]-peaks[20]
for i in range(0,3):
	print('Gap13',(i+1),': ', Gap13[i]/1000,'(/2pi)kHz')


print('10mm')
f,A = np.genfromtxt('FP-V23data/4.3_400mm_10mm.dat',unpack=True)
indices = find_peaks(A,prominence=2)
peaks = f[indices[0][:7]]
indices = find_peaks(A,prominence=0.5)
peaks = np.append(peaks,f[indices[0][9:]])
#print('peaks:', peaks)
peaks = unp.uarray(peaks,10)*2*np.pi

n = np.linspace(2,8,7)
n = np.append(n,np.linspace(10,16,7))
n = np.append(n,np.linspace(18,23,6))
n = np.append(n,np.linspace(25,30,5))
k = (n*np.pi)/0.4
#print('k:', k)

numSegments = 8
L = 0.4
a = L/numSegments
PlotBandStructure(k,peaks,'4.3_10mm',type='extended')
PlotBandStructure(k,peaks,'4.3_10mm',type='reduced',a=a,numBands=4)

Band10 = unp.uarray([0,0,0,0],[0,0,0,0])
Band10[0] = peaks[6]-peaks[0]
Band10[1] = peaks[13]-peaks[7]
Band10[2] = peaks[19]-peaks[14]
Band10[3] = peaks[24]-peaks[20]
for i in range(0,4):
	print('Band10',(i+1),': ', Band10[i]/1000,'(/2pi)kHz')

Gap10 = unp.uarray([0,0,0],[0,0,0])
Gap10[0] = peaks[7]-peaks[6]
Gap10[1] = peaks[14]-peaks[13]
Gap10[2] = peaks[20]-peaks[19]
for i in range(0,3):
	print('Gap10',(i+1),': ', Gap10[i]/1000,'(/2pi)kHz')


#4.6
print(' ')
print('4.6')

f,A = np.genfromtxt('FP-V23data/4.6_50mm.dat',unpack=True)
indices = find_peaks(A,20,prominence=2)
peaks = f[indices[0]]
peaks = peaks[2:6]
print('peaks:', peaks)

Df = []
for i in range(len(peaks)-1):
	temp = peaks[i+1]-peaks[i]
	Df = Df+[temp]
Df = avg_and_sem(Df)
Df = unp.uarray(Df[0],Df[1])
print('Df:', Df,'Hz')


#4.7
print(' ')
print('4.7')

f,A = np.genfromtxt('FP-V23data/4.7_75mm.dat',unpack=True)
indices = find_peaks(A,15,prominence=2)
peaks = f[indices[0]]
peaks = peaks[1:7]
print('peaks:', peaks)

Df = []
for i in range(len(peaks)-1):
	temp = peaks[i+1]-peaks[i]
	Df = Df+[temp]
Df = avg_and_sem(Df)
Df = unp.uarray(Df[0],Df[1])
print('Df:', Df,'Hz')


#4.10
print(' ')
print('4.10')

print('4.10.1')
f,A = np.genfromtxt('FP-V23data/4.4_600mm_16mm.dat',unpack=True)
indices = find_peaks(A,prominence=0.1)
peaks = f[indices[0]]
#print('peaks:', peaks)
peaks = unp.uarray(peaks,10)*2*np.pi

n = np.linspace(2,24,23)
n = np.append(n,np.linspace(26,35,10))
n = np.append(n,np.linspace(38,47,10))
k = (n*np.pi)/0.6
#print('k:', k)

Plot(k,peaks,[0,130],r'$k/\si{\per\metre}$',r'$\omega/\si{\kilo\hertz}$',params,'4.12(4.4_600mm_16mm)',linear=False,yscale=1/1000,limy=[0,35000])

numSegments = 12
L = 0.6
a = L/numSegments
PlotBandStructure(k,peaks,'4.10(4.4_600mm_16mm)',type='extended')
PlotBandStructure(k,peaks,'4.10(4.4_600mm_16mm)',type='reduced',a=a,numBands=4)

peaks = peaks[:23]
k = k[:23]
PlotBandStructure(k,peaks,'4.12(4.4_600mm_16mm)',type='extended')
PlotBandStructure(k,peaks,'4.12(4.4_600mm_16mm)',type='reduced',a=a,numBands=2)


print('4.10.2')
f,A = np.genfromtxt('FP-V23data/4.10_600mm_13_16mm.dat',unpack=True)
indices = find_peaks(A,prominence=0.1)
peaks = f[indices[0]]
peaks = peaks[1:]
#print('peaks:', peaks)
peaks = unp.uarray(peaks,10)*2*np.pi

n = np.linspace(2,23,22)
n = np.append(n,np.linspace(26,30,5))
n = np.append(n,np.linspace(32,35,4))
n = np.append(n,np.linspace(38,42,5))
n = np.append(n,np.linspace(44,46,3))
k = (n*np.pi)/0.6
#print('k:', k)

numSegments = 12
L = 0.6
a = L/numSegments
PlotBandStructure(k,peaks,'4.10',type='extended')
PlotBandStructure(k,peaks,'4.10',type='reduced',a=a,numBands=4)


print('4.11')
f,A = np.genfromtxt('FP-V23data/4.11_625mm_50_16_75mm.dat',unpack=True)
indices = find_peaks(A,prominence=0.1)
peaks = f[indices[0]]
#print('peaks:', peaks)
peaks = unp.uarray(peaks,10)*2*np.pi

n = np.linspace(3,34,32)
n = np.append(n,np.linspace(37,39,3))
n = np.append(n,np.linspace(42,44,3))
n = np.append(n,np.linspace(47,49,3))
k = (n*np.pi)/0.625
#print('k:', k)

numSegments = 5
L = 0.625
a = L/numSegments
marker = ['c-','b-','c-','b-','c-','b-','c-','b-','c-','b-']
ranges = [0,8,5,5,5,9,3,3,3]
PlotBandStructure(k,peaks,'4.11',type='extended')
PlotBandStructure(k,peaks,'4.11',type='reduced',a=a,numBands=10,marker=marker,ranges=ranges)


print('4.12')
f,A = np.genfromtxt('FP-V23data/4.12_seg3_75mm.dat',unpack=True)
indices = find_peaks(A,prominence=0.5)
peaks = f[indices[0]]
#print('peaks:', peaks)
peaks = unp.uarray(peaks,10)*2*np.pi

n = np.linspace(2,24,23)
k = (n*np.pi)/0.625
#print('k:', k)

numSegments = 12
L = 0.625
a = L/numSegments
PlotBandStructure(k,peaks,'4.12',type='extended')
PlotBandStructure(k,peaks,'4.12',type='reduced',a=a,numBands=2)
