from table import makeTable
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp

#Temparaturgraphen
T1, T2, Leistung, Pa, Pb = np.genfromtxt("scripts/data1.txt", unpack=True)
Pa = Pa+1
Pa = Pa *100000
Pb = Pb+1
Pb = Pb *100000
T1 += 273.15
T2 += 273.15
Zeitab = np.linspace(60,1080,len(T1))
#Konstanten
cWasser = 4183#j/(kg*K)


#Die einzelnen Approximationskurven im Vergleich
def Polynom(x, A, B, C):
    return A*x*x+B*x+C
	
def linear(x, A, B):
    return A*x+B

#Fehler
paramsPolynomT1, covariancePolynomT1 = curve_fit(Polynom, Zeitab, T1)
paramsPolynomT2, covariancePolynomT2 = curve_fit(Polynom, Zeitab, T2)
errorsPT1 = np.sqrt(np.diag(covariancePolynomT1))
errorsPT2 = np.sqrt(np.diag(covariancePolynomT2))

print('Polynomapproximation T1')
print('A =', paramsPolynomT1[0], 'pm', errorsPT1[0])
print('B =', paramsPolynomT1[1], 'pm', errorsPT1[1])
print('C =', paramsPolynomT1[2], 'pm', errorsPT1[2])
print('Polynomapproximation T2')
print('A =', paramsPolynomT2[0], 'pm', errorsPT2[0])
print('B =', paramsPolynomT2[1], 'pm', errorsPT2[1])
print('C =', paramsPolynomT2[2], 'pm', errorsPT2[2])

#Fehlerarrays
PolynomAT1 = unp.uarray(paramsPolynomT1[0], errorsPT1[0])
PolynomBT1 = unp.uarray(paramsPolynomT1[1], errorsPT1[1])
PolynomAT2 = unp.uarray(paramsPolynomT2[0], errorsPT2[0])
PolynomBT2 = unp.uarray(paramsPolynomT2[1], errorsPT1[1])

plt.cla()
plt.clf()
x_plot = np.linspace(0, 1140)
#Graphenapproximation
#Graphmit Temperaturen
plt.plot(Zeitab, T1, 'rx', label ="Temperatur von Reservoir 1")
plt.plot(Zeitab, T2, 'gx', label = "Temperatur von Reservoir 2")
plt.xlim(0, 1140)
plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$T / \si{\kelvin}$')
plt.legend(loc="best")
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig("build/Temperaturen")

plt.cla()
plt.clf()
#Fitgraph von T1
plt.plot(Zeitab, T1, 'rx', label ="Temperatur von Reservoir 1")
plt.plot(x_plot, Polynom(x_plot, *paramsPolynomT1), 'b-', label='Fit durch Polynom 2. Grades')
plt.xlim(0, 1140)
plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$T / \si{\kelvin}$')
plt.legend(loc="best")
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig("build/T1")

plt.cla()
plt.clf()
#Graphtemparatur2
plt.plot(Zeitab, T2, 'gx', label = "Temperatur von Reservoir 2")
plt.plot(x_plot, Polynom(x_plot, *paramsPolynomT2), 'b-', label='Fit durch Polynom 2. Grades')
plt.xlim(0, 1140)
plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$T / \si{\kelvin}$')
plt.legend(loc="best")
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig("build/T2")


#Güte bestimmen
#Formel
A2T1 = 2*PolynomAT1
A2T2 = 2*PolynomAT2
print('koeffableitungAT1',A2T1)
print('koeffableitungAT2',A2T2)

def Ableitung(x, A, B):
    return A*x+B


#print('AbleitungT1',Ableitung(12*60,A2T1,PolynomBT1))
#print('AbleitungT2',Ableitung(12*60,A2T2,PolynomBT2))
def realGuete(Jim,m1,mkck,cw,N):
    return (m1*cw+mkck)*Jim/N

print('GueteT1bei 4', realGuete(Ableitung(Zeitab[4], A2T1, PolynomBT1), 3, 660, cWasser, Leistung[4]))
print('GueteT1bei 8', realGuete(Ableitung(Zeitab[8], A2T1, PolynomBT1), 3,660, cWasser, Leistung[8]))
print('GueteT1bei 12', realGuete(Ableitung(Zeitab[12], A2T1, PolynomBT1),3,660,cWasser,Leistung[12]))
print('GueteT1bei 16', realGuete(Ableitung(Zeitab[16], A2T1, PolynomBT1),3,660,cWasser,Leistung[16]))

Ableitungen = [unp.nominal_values(Ableitung(Zeitab[4],A2T1,PolynomBT1)),
unp.nominal_values(Ableitung(Zeitab[8], A2T1, PolynomBT1)),
unp.nominal_values(Ableitung(Zeitab[12], A2T1, PolynomBT1)),
unp.nominal_values(Ableitung(Zeitab[16], A2T1, PolynomBT1))]


Ableitungenfe =[unp.std_devs(Ableitung(Zeitab[4],A2T1,PolynomBT1)),
unp.std_devs(Ableitung(Zeitab[8],A2T1,PolynomBT1)),
unp.std_devs(Ableitung(Zeitab[12],A2T1,PolynomBT1)),
unp.std_devs(Ableitung(Zeitab[16],A2T1,PolynomBT1))]
Ableitungen =np.array(Ableitungen)
Ableitungenfe =np.array(Ableitungenfe)



Ableitungen2 = [unp.nominal_values(Ableitung(Zeitab[4],A2T2,PolynomBT2)),
unp.nominal_values(Ableitung(Zeitab[8], A2T2, PolynomBT2)),
unp.nominal_values(Ableitung(Zeitab[12], A2T2, PolynomBT2)),
unp.nominal_values(Ableitung(Zeitab[16], A2T2, PolynomBT2))]


Ableitungenfe2 =[unp.std_devs(Ableitung(Zeitab[4],A2T2,PolynomBT2)),
unp.std_devs(Ableitung(Zeitab[8],A2T2,PolynomBT2)),
unp.std_devs(Ableitung(Zeitab[12],A2T2,PolynomBT2)),
unp.std_devs(Ableitung(Zeitab[16],A2T2,PolynomBT2))]
Ableitungen2 =np.array(Ableitungen2)
Ableitungenfe2 =np.array(Ableitungenfe2)





guete = [unp.nominal_values(realGuete(Ableitung(Zeitab[4],A2T1,PolynomBT1),3,660,cWasser,Leistung[4])),
unp.nominal_values(realGuete(Ableitung(Zeitab[8],A2T1,PolynomBT1),3,660,cWasser,Leistung[8])),
unp.nominal_values(realGuete(Ableitung(Zeitab[12],A2T1,PolynomBT1),3,660,cWasser,Leistung[12])),
unp.nominal_values(realGuete(Ableitung(Zeitab[16],A2T1,PolynomBT1),3,660,cWasser,Leistung[16]))]


guetefehler = [unp.std_devs(realGuete(Ableitung(Zeitab[4],A2T1,PolynomBT1),3,660,cWasser,Leistung[4])),
unp.std_devs(realGuete(Ableitung(Zeitab[8],A2T1,PolynomBT1),3,660,cWasser,Leistung[8])),
unp.std_devs(realGuete(Ableitung(Zeitab[12],A2T1,PolynomBT1),3,660,cWasser,Leistung[12])),
unp.std_devs(realGuete(Ableitung(Zeitab[16],A2T1,PolynomBT1),3,660,cWasser,Leistung[16]))]


idealguete = [T1[4]/(T1[4]-T2[4]),T1[8]/(T1[8]-T2[8]),T1[12]/(T1[12]-T2[12]),T1[16]/(T1[16]-T2[16])]
#ideal
print(T1[4]/(T1[4]-T2[4]))
print(T1[8]/(T1[8]-T2[8]))
print(T1[12]/(T1[12]-T2[12]))
print(T1[16]/(T1[16]-T2[16]))
#Rechnung

#Dampfdruckkurve L-Bestimmung
Dampfdruck, covarianceDampfdruck = curve_fit(linear,1/T1 , np.log(Pb))
errorsDampfdruck = np.sqrt(np.diag(covarianceDampfdruck))
DampfdruckA = unp.uarray(Dampfdruck[0], errorsDampfdruck[0])
DampfdruckB = unp.uarray(Dampfdruck[1], errorsDampfdruck[1])
#Graph
plt.cla()
plt.clf()
Dampf_plot = 1/np.linspace(273.15+20, 273.15+55)
plt.plot(1/T1*1000, np.log(Pb), 'rx', label ="Druck gegen Temperatur")
plt.plot(Dampf_plot*1000, linear(Dampf_plot, *Dampfdruck), 'b-', label='linearer Fit')
plt.xlim(0.00305*1000, 0.003405*1000)
plt.xlabel(r'$T^{-1}/\si[per-mode=reciprocal]{\per\kilo\kelvin}$')
plt.ylabel(r'$\ln\left(p\right)/\si{\pascal}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc="best")
plt.savefig("build/Dampdruck")

print('Dampdrucksteigung',Dampfdruck[0],'pm',errorsDampfdruck[0])
print('Achsenabschnitt',Dampfdruck[1],'pm',errorsDampfdruck[1])
m=unp.uarray(Dampfdruck[0], errorsDampfdruck[0])
R = unp.uarray(8.3144598, 0.0000048)
L = -m*R
print('L:',L)

#Massendruchsatz
def Massendurch(Jim2,m2,cw,mkck,L):
    return (m2*cw+mkck)*Jim2/L

UDampfdruck = -unp.uarray(Dampfdruck[0],errorsDampfdruck[0])



print('Massendurchsatz4',Massendurch(Ableitung(4*60,A2T2,PolynomBT2),3,cWasser,660,-R*1000/18*UDampfdruck))
print('Massendurchsatz8',Massendurch(Ableitung(8*60,A2T2,PolynomBT2),3,cWasser,660,-R*1000/18*UDampfdruck))
print('Massendurchsatz12',Massendurch(Ableitung(12*60,A2T2,PolynomBT2),3,cWasser,660,-R*1000/18*UDampfdruck))
print('Massendurchsatz16',Massendurch(Ableitung(16*60,A2T2,PolynomBT2),3,cWasser,660,-R*1000/18*UDampfdruck))

massen =[unp.nominal_values(Massendurch(Ableitung(4*60,A2T2,PolynomBT2),3,cWasser,660,-R*1000/18*UDampfdruck)),
unp.nominal_values(Massendurch(Ableitung(8*60,A2T2,PolynomBT2),3,cWasser,660,-R*1000/18*UDampfdruck)),
unp.nominal_values(Massendurch(Ableitung(12*60,A2T2,PolynomBT2),3,cWasser,660,-R*1000/18*UDampfdruck)),
unp.nominal_values(Massendurch(Ableitung(16*60,A2T2,PolynomBT2),3,cWasser,660,-R*1000/18*UDampfdruck))]


massenfehler =[unp.std_devs(Massendurch(Ableitung(4*60,A2T2,PolynomBT2),3,cWasser,660,-R*1000/18*UDampfdruck)),
unp.std_devs(Massendurch(Ableitung(8*60,A2T2,PolynomBT2),3,cWasser,660,-R*1000/18*UDampfdruck)),
unp.std_devs(Massendurch(Ableitung(12*60,A2T2,PolynomBT2),3,cWasser,660,-R*1000/18*UDampfdruck)),
unp.std_devs(Massendurch(Ableitung(16*60,A2T2,PolynomBT2),3,cWasser,660,-R*1000/18*UDampfdruck))]

massen = np.array(massen)
massenfehler = np.array(massenfehler)
Umassen = unp.uarray(massen, massenfehler)
zeiten = [4*60, 8*60, 12*60, 16*60]
#
makeTable([T1-273.15, T2-273.15, Pa/100000, Pb/100000, Leistung], r'{$T_1 \si{\degreeCelsius}$} & {$T_2 \si{\degreeCelsius}$} & {$p_\text{a}/\si{\bar}$} & {$p_\text{b}/\si{\bar}$} & {$N_\text{mech}/\si{\watt}$}', 'tabges', ['S[table-format=2.1]', 'S[table-format=2.1]', 'S[table-format=1.2]', 'S[table-format=2.2]', 'S[table-format=3.0]'], ["%2.1f", "%2.1f", "%1.2f", "%2.2f", "%3.0f"])

makeTable([zeiten, guete,guetefehler,idealguete], r'{'+r't/\si{\second}'+r'} & \multicolumn{2}{c}{'+r'$v_\text{real}$'+r'} & {'+r'$v_\text{ideal}$'+r'}', 'tabv', ['S[table-format=3.0]', 'S[table-format=1.1]', ' @{${}\pm{}$} S[table-format=1.1]', 'S[table-format=2.1]'], ["%3.0f", "%1.1f", "%1.1f", "%2.1f"])

makeTable([zeiten, massen*1000,massenfehler*1000], r'{'+r't/\si{\second}'+r'} & \multicolumn{2}{c}{'+r'$\frac{\text{d}m}{\text{d}t}/\si[per-mode=reciprocal]{\gram\per\second}$'+r'}', 'tabm', ['S[table-format=3.0]', 'S[table-format=1.2]', ' @{${}\pm{}$} S[table-format=1.2]'], ["%3.0f", "%1.2f", "%1.2f"])

def dichte(Pa,T2):
    return(Pa*5.51*273.15)/(100000*T2)

#Leistung
def Nmech(k, Pa,Pb,delm,roh):
    return (1/(k-1))*(delm/roh)*(Pb*((Pa/Pb)**(1/k))-Pa)

N1 = unp.nominal_values(Nmech(1.14,Pa[4],Pb[4],Umassen[0],dichte(Pa[4],T2[4])))
N2 = unp.nominal_values(Nmech(1.14,Pa[8],Pb[8],Umassen[1],dichte(Pa[8],T2[8])))
N3 = unp.nominal_values(Nmech(1.14,Pa[12],Pb[12],Umassen[2],dichte(Pa[12],T2[12])))
N4 = unp.nominal_values(Nmech(1.14,Pa[16],Pb[16],Umassen[3],dichte(Pa[16],T2[16])))
N1fe = unp.std_devs(Nmech(1.14,Pa[4],Pb[4],Umassen[0],dichte(Pa[4],T2[4])))
N2fe = unp.std_devs(Nmech(1.14,Pa[8],Pb[8],Umassen[1],dichte(Pa[8],T2[8])))
N3fe = unp.std_devs(Nmech(1.14,Pa[12],Pb[12],Umassen[2],dichte(Pa[12],T2[12])))
N4fe = unp.std_devs(Nmech(1.14,Pa[16],Pb[16],Umassen[3],dichte(Pa[16],T2[16])))


Narray = [N1,N2,N3,N4]
Narray = np.array(Narray)
Nfearray = [N1fe,N2fe,N3fe,N4fe]
Nfearray = np.array(Nfearray)
print(Narray)
print(Nfearray)
makeTable([zeiten, Narray,Nfearray], r'{'+r't/\si{\second}'+r'} & \multicolumn{2}{c}{'+r'$N_\text{mech}\si{\watt}$'+r'}', 'tabn', ['S[table-format=3.0]', 'S[table-format=1.1]', ' @{${}\pm{}$} S[table-format=1.1]'], ["%3.0f", "%1.1f", "%1.1f"])


makeTable([zeiten, Ableitungen,Ableitungenfe, Ableitungen2, Ableitungenfe2], r'{'+r't/\si{\second}'+r'} & \multicolumn{2}{c}{'+r'$\frac{\text{d}T_1}{\text{d}t}/\si[per-mode=reciprocal]{\kelvin\per\second}$'+r'} & \multicolumn{2}{c}{'+r'$\frac{\text{d}T_2}{\text{d}t}/\si[per-mode=reciprocal]{\kelvin\per\second}$'+r'}', 'taba', ['S[table-format=2.0]', 'S[table-format=2.3]', ' @{${}\pm{}$} S[table-format=1.3]','S[table-format=2.3]', ' @{${}\pm{}$} S[table-format=1.3]'], ["%2.0f", "%2.3f", "%2.3f", "%2.3f", "%2.3f"])
