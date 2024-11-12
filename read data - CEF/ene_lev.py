import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
import matplotlib
matplotlib.use('QtAgg')

def cry_fie_cal(Bdictionary):
    ion = 'Tb3+'
    TVS = cef.CFLevels.Bdict(ion, Bdictionary)
    TVS.diagonalize()
    print('********************************************')
    print('Eigenvalues of Hamiltonian based on provided B:\n{}'.format(TVS.eigenvalues))
    TVS.printEigenvectors()
    return TVS

B20 = -0.102575
B40 = -0.000590838
B60 = 2.24021e-6
# B66List=[0]
B66=2.5e-6
# B20 = -0.1114
# B40 = -0.000542
# B60 = 1.92e-6
# # B66List=[0]
# B66=3e-5

Bdictionary = {'B20': B20, 'B40': B40,
               'B60': B60, 'B66': B66}
# #
TVS = cry_fie_cal(Bdictionary)
print(TVS.eigenvalues)

fig,ax = plt.subplots()
x = np.linspace(0,1,1000)
y = np.ones(1000)
for idx,eng in enumerate(TVS.eigenvalues):
    ax.plot(x,y*TVS.eigenvalues[idx], label=eng)
ax.get_xaxis().set_visible(False)
ax.tick_params(axis='y', which='major', labelsize=14)
ax.set_xlim(left=0,right=1.4)
ax.set_ylabel('Energy (meV)', fontsize=14)

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.show()