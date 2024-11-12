import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef


ion = 'Tb3+'
B20=-0.0
B40=-0.0007
B60 = 0.0
B66 = -8e-6

Bdictionary = {'B20': B20, 'B40': B40,
			   'B60': B60, 'B66': B66}
TVS = cef.CFLevels.Bdict(ion, Bdictionary)
TVS.diagonalize()
print(TVS.eigenvalues)

# Tb3_SOC = -1705/8.066 ## from  <https://doi.org/10.1063/1.1701548> ??
Bdict = {'B20': B20* cef.LSThet[ion][0]/cef.Thet[ion][0],
         'B40': B40*cef.LSThet[ion][1]/cef.Thet[ion][1],
		 'B60': B66*cef.LSThet[ion][2]/cef.Thet[ion][2],
	 'B66': B66*cef.LSThet[ion][2]/cef.Thet[ion][2]}
Tb2 = cef.LS_CFLevels.Bdict(Bdict,3,3,-100000.)
Tb2.diagonalize()
print(Tb2.eigenvalues)



xx = np.linspace(0,3,1000)
plt.figure()
yy = TVS.normalizedNeutronSpectrum(Earray=xx, Temp=8, ResFunc=lambda x: 0.1)
plt.plot(xx,yy,'-')
plt.xlabel('$\hbar \omega$ (meV)')
plt.ylabel('Intensity (a.u.)')
plt.show()