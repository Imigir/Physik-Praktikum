import numpy as np
import math

L=10.11/1000
C=2.098/1000000000
Lerr=0.03/1000
Cerr=0.006/1000000000
R=2*np.sqrt(L/C)
fehR=np.sqrt(Lerr**2/(C*L)+(L*Cerr)**2/(C**3*L))
print('R_ap =',R,'+-',fehR)
print((R-3500)/3500)
