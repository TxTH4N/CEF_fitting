import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
import scipy.linalg as LA
from pcf_lib.Operators import Ket, Operator, LSOperator

g = 3/2
mu_b = 0.05788
Field = [0,0,1.5]

ion = 'Tb3+'
B20=-0.1036
B40=-5.91e-4
B60 = 2.31e-6
B66 = 3.3e-6

Bdictionary = {'B20': B20, 'B40': B40,
			   'B60': B60, 'B66': B66}
TVS = cef.CFLevels.Bdict(ion, Bdictionary)
TVS.diagonalize()
Jx = TVS.opttran.Jx
Jy = TVS.opttran.Jy * 1j
Jz = TVS.opttran.Jz
JdotB = g*mu_b*(Field[0]*Jx + Field[1]*Jy + Field[2]*Jz)

FieldHam = TVS.H + JdotB
diagonalH = LA.eigh(FieldHam)

minE = np.amin(diagonalH[0])
evals = diagonalH[0] - minE
evecs = diagonalH[1].T
# print(evals)
# print(evecs)
JexpVals = np.zeros((len(evals), 3))
for i, ev in enumerate(evecs):
	kev = Ket(ev)
	# print np.real(np.dot(ev,kev.Jy().ket)), np.real(np.dot(ev,np.dot(Jy.O,ev)))
	# print np.real(kev*kev.Jy()) - np.real(np.dot(ev,np.dot(Jy.O,ev)))
	JexpVals[i] = [np.real(kev * kev.Jx()),
				   np.real(kev * kev.Jy()),
				   np.real(kev * kev.Jz())]
print(JexpVals)

# # Tb3_SOC = -1705/8.066 ## from  <https://doi.org/10.1063/1.1701548> ??
# Bdict = {'B20': B20* cef.LSThet[ion][0]/cef.Thet[ion][0],
#          'B40': B40*cef.LSThet[ion][1]/cef.Thet[ion][1],
# 		 'B60': B66*cef.LSThet[ion][2]/cef.Thet[ion][2],
# 	 'B66': B66*cef.LSThet[ion][2]/cef.Thet[ion][2]}
# Tb2 = cef.LS_CFLevels.Bdict(Bdict,3,3,-100000.)
# Tb2.diagonalize()
# print(Tb2.eigenvalues)



xx = np.linspace(0,12,1000)
plt.figure()
yy = TVS.normalizedNeutronSpectrum(Earray=xx, Temp=8, ResFunc=lambda x: 0.1)
plt.plot(xx,yy,'-')
plt.xlabel('$\hbar \omega$ (meV)')
plt.ylabel('Intensity (a.u.)')
plt.show()