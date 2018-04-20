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

t,N = np.genfromtxt('data1.txt', unpack = True)
N_0 = 223/900
N_err = np.sqrt(N)/30
N = N/30 - N_0
N = np.log(N)
N = unp.uarray(N, N_err)
t = t*30

makeTable([np.exp(noms(N)*30), stds(N)*30, noms(N), stds(N), t, noms(N)*np.log(stds(N))-noms(N), noms(N)-noms(N)/np.log(stds(N)), r'\multicolumn{2}{c}{'+r'$N_.{exp}/\si{1\per30\second}$'+r'} &\multicolumn{2}{c}{'+r'$\ln{N)}/\si{\becquerel}$'+r'} & {'+r'$t/\si{\second}$'+r'} & {'+r'$\ln{(N+\sigma)}-\ln{(N)}/\si{\becquerel}$'+r'} &{'+r'$\ln{(N)}-\ln{(N-\sigma)}/\si{\becquerel}$'+r'},'tab1', ['S[table-format=5.0]', ' @{${}\pm{}$} S[table-format=3.0]', 'S[table-format=3.0]', ' @{${}\pm{}$} S[table-format=1.2]', 'S[table-format=3.0]', 'S[table-format=1.2]', 'S[table-format=1.2]', ["%5.0f", "%3.0f", "%3.0f", "%1.0f", "%3.0f", "%1.2f", "%3.1f", "%1.1f"])
