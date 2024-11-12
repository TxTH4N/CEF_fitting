import numpy as np
import matplotlib as plt
import PyCrystalField as cef

RTOLig, R = cef.importCIF('RTO.cif', 'Ho')
# R.printEigenvectors()

# #
#
# the2 = -1/(2*3**2*5**2)
# the4 = -1/(2*3*5*7*11*13)
# the6 = -5/(3**3*7*11**2*13**2)
#
# l20 = 1/2
# l40 = 1/8
# l43 = 0.5*np.sqrt(35)
# l60 = 1/16
# l63 = 1/8*np.sqrt(105)
# l66 = 1/16*np.sqrt(231)
#
# B20 =68.2*l20*the2
# B40 = 274.8*l40*the4
# B43 = 83.7*l43*the4
# B60 = 86.8*l60*the6
# B63 = -62.5*l63*the6
# B66 = 101.6*l66*the6
# #
# Bdictionary = {'B20': B20, 'B40': B40,'B43': B43,
#                'B60': B60, 'B63': B63, 'B66': B66}
# #
# ion = 'Ho3+'
# BTO = cef.CFLevels.Bdict(ion, Bdictionary)
# BTO.diagonalize()
# print('********************************************')
# print('Eigenvalues of Hamiltonian based on provided B:\n{}'.format(BTO.eigenvalues))
# BTO.printEigenvectors()

# [-0.07577777777777778, -0.001143856143856144, -0.008244686610813151, -7.02e-06,       1.035e-4,   -1.2485e-4]
# [-0.048269,            -0.00099833,           -0.00735134,            -3.68e-06,      3.307e-05,  -3.467e-05]

print(RTOLig.ligandPos)