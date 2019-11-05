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
import uncertainties 
from uncertainties import ufloat
import scipy.constants as const
from errorfunkt2tex import error_to_tex
from errorfunkt2tex import scipy_to_unp
from sympy import *
import scipy.constants as const


#Daten
I, B  = np.genfromtxt('scripts/BFeldKali.txt',unpack=True)
B = B/1000

#x werte nun in mm
print("I in ampere:", I)
print("B in tesla:", B)
#hier tabellen erzeugen
makeTable([I,B], r'{$ I / \si{\ampere}$} & {$ B/ \si{\tesla}$}','tabIB1' , ['S[table-format=2.1]' , 'S[table-format=1.3]'] ,  ["%2.1f", "%1.3f"])
#makeTable([I[:],B[:]], r'{$ I/ \si{\ampere}$} & {$ B/ \si{\tesla}$}','tabIB2' , ['S[table-format=2.1]' , 'S[table-format=1.3]'] ,  ["%2.1f", "%1.3f"])

#bfeld kalibrierung


def BvonI(x,a,b):
	return a*x+b
	

params, covariance_matrix = curve_fit(BvonI,I,B)
#errors = unp.uarray(params, np.sqrt(np.diag(covariance_matrix)))
errors = uncertainties.correlated_values(params, covariance_matrix)
print('Die Parameter der B feld kalibrierung:')
print('a =' , errors[0])
print('b =',  errors[1])
tesla = errors

#der plot
x = np.linspace(0, I[-1],10000)
plt.cla()
plt.clf()
plt.plot(x, BvonI(x,unp.nominal_values(tesla[0]),unp.nominal_values(tesla[1])), 'b-', label='Ausgleichsgerade')
plt.plot(I, B, 'rx', label='Messdaten')
#plt.ylim(0, line(t[-1], *params)+0.1)
#plt.xlim(0, t[-1]*100)
plt.xlabel(r'$I/\si{\ampere}$')
plt.ylabel(r'$B /\si{\tesla} $')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/'+'Bfeldkali')

#konstanten
lambdan = ufloat(643.8*10**(-9),0)
lambdaa = ufloat(480*10**(-9),0)
lambdaDn = 4.891*10**(-11)
lambdaDa = 2.695*10**(-11)
h = const.h
clight = const.c
ub = const.physical_constants["Bohr magneton"][0]


#normaler Zeemaneffekt
delta_sn, sigma_sn  = np.genfromtxt('scripts/rot.txt',unpack=True)
#ma1 und ma2 müssen noch in die datei

#berechne delta lambda
def sigmalambda(delta_s,sigma_s,deltalambda):
	return 0.5*(sigma_s / delta_s)*deltalambda

sigmalambdan = sigmalambda(delta_sn,sigma_sn,lambdaDn)
print('sigmalambdan:', sigmalambdan)

#mittlere sigmalambda
sigmalambdanm = avg_and_sem(sigmalambdan)
sigmalambdanm = unp.uarray(sigmalambdanm[0],sigmalambdanm[1])
print('sigmalambdanm =' , sigmalambdanm)

#berechne g faktor von nomalem zeeman
g = (h*clight/lambdan**2)*sigmalambdanm*1/(ub* BvonI(10.5,tesla[0],tesla[1]))
print('g =' , g)
print('B(10.5ampere) =',BvonI(10.5,tesla[0],tesla[1]))
#mögliche ursache: b feld kalibrierung



#annormaler zeemaneffekt sigmalinien
delta_sa, sigma_sa1, sigma_sa2 = np.genfromtxt('scripts/blau.txt',unpack=True)

# sigma-linien
sigmalambdaa1 = sigmalambda(delta_sa,sigma_sa1,lambdaDa)
print('sigmalambdaa1:', sigmalambdaa1)

#mittlere sigmalambda
sigmalambdaa1m = avg_and_sem(sigmalambdaa1)
sigmalambdaa1m = unp.uarray(sigmalambdaa1m[0],sigmalambdaa1m[1])
print('sigmalambdaa1m =', sigmalambdaa1m)

#berechne g faktor von anomalem zeeman
ga1 = (h*clight/lambdaa**2)*sigmalambdaa1m*1/(ub* BvonI(5.5,tesla[0],tesla[1]))
print('ga1 =' , ga1)
print('B(5.5ampere) =',BvonI(5.5,tesla[0],tesla[1]))

#pi-linien
sigmalambdaa2 = sigmalambda(delta_sa,sigma_sa2,lambdaDa)
print('sigmalambdaa2:', sigmalambdaa2)

#mittlere sigmalambda
sigmalambdaa2m = avg_and_sem(sigmalambdaa2)
sigmalambdaa2m = unp.uarray(sigmalambdaa2m[0],sigmalambdaa2m[1])
print('sigmalambdaa2m =', sigmalambdaa2m)

#berechne g faktor von anomalem zeeman
ga2 = (h*clight/lambdaa**2)*sigmalambdaa2m*1/(ub* BvonI(18,tesla[0],tesla[1]))
print('ga2 =' , ga2)
print('B(18ampere) =',BvonI(18,tesla[0],tesla[1]))

"""
#tabellen
makeTable([delta_sn,delta_sa1,delta_sa2], r'{$ \Delta s_{\text{norm},\sigma}/ px$} & {$ \Delta s_{\text{anorm},\sigma/}/ px$} & {$ \Delta s_{\text{anorm},\pi}/ px$}','tabdelta' , ['S[table-format=3.0]' , 'S[table-format=3.0]' , 'S[table-format=3.0]'] ,  ["%3.0f", "%3.0f", "%3.0f"])
makeTable([sigma_sn,sigma_sa1,sigma_sa2], r'{$ \delta s_{\text{norm},\sigma} / px$} & {$ \delta s_{\text{anorm},\sigma}/ px $} & {$  \delta s_{\text{anorm},\pi}/ px $}','tabsigma' , ['S[table-format=3.0]' , 'S[table-format=3.0]' , 'S[table-format=3.0]'] ,  ["%3.0f", "%3.0f", "%3.0f"])
makeTable([sigmalambdan*10**12,sigmalambdaa1*10**12,sigmalambdaa2*10**12], r'{$ \delta\lambda s_{\text{norm},\sigma}/ \si{\pico\meter}$} & {$  \delta\lambda s_{\text{anorm},\sigma}/ \si{\pico\meter}$} & {$  \delta\lambda s_{\text{norm},\pi}/ \si{\pico\meter}$}','deltalambda' , ['S[table-format=2.1]' , 'S[table-format=2.1]' , 'S[table-format=2.1]'] ,  ["%2.1f", "%2.1f", "%2.1f"])
"""

