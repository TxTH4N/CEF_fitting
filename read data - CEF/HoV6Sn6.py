import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef

def ins_res(x, ei=3.32):
    if ei == 3.32:
        y = +0.00030354 * x ** 3 + 0.0039009 * x ** 2 - 0.040862 * x + 0.11303
    elif ei == 12:
        y = 75e-05 * x ** 3 + 0.0018964 * x ** 2 - 0.078275 * x + 0.72305
    return y

# Tb=[-1.02669522e-01, -5.93838489e-04,  2.32111476e-06,  4.89792347e-12]
# Tb = [-0.102, -5.95e-4,2.26e-6,2.5e-6]
Tb = [-0.1114, -5.42e-4,1.92e-6,3e-5]
# Tb = [-1.11601415e-01,-5.58310140e-04,2.16914592e-06,2.12931712e-05]

Bdictionary = {'B20': Tb[0], 'B40': Tb[1],
               'B60': Tb[2], 'B66': Tb[3]}
TVS = cef.CFLevels.Bdict('Tb3+', Bdictionary)
TVS.diagonalize()
print('Eigenvalues of Hamiltonian based on provided B:\n{}'.format(TVS.eigenvalues))


alpha_Tb = -1/(3**2*11)
beta_Tb = 2/(3**3*5*11**2)
gamma_Tb = -1/(3**4*7*11**2*13)

alpha_Ho = -1/(2*3**2*5**2)
beta_Ho = -1/(2*3*5*7*11*13)
gamma_Ho = -5/(3**3*7*11**2*13**2)
ion = 'Ho3+'
B20 = Tb[0]*alpha_Ho/alpha_Tb
B40 = Tb[1]*beta_Ho/beta_Tb
B60 = Tb[2]*gamma_Ho/gamma_Tb
B66 =Tb[3]*gamma_Ho/gamma_Tb

Bdictionary = {'B20': B20, 'B40': B40,
               'B60': B60, 'B66': B66}
#
HVS = cef.CFLevels.Bdict(ion, Bdictionary)
HVS.diagonalize()
print('Eigenvalues of Hamiltonian based on provided B:\n{}'.format(HVS.eigenvalues))
HVS.printEigenvectors()
# HVS.printLaTexEigenvectors()


x_e_long = np.linspace(0, 20, 1000)
fig, ax = plt.subplots()
sim_int_2 = 0.0005 * HVS.normalizedNeutronSpectrum(x_e_long, Temp=0.1, ResFunc= lambda x : 0.1)
ax.plot(x_e_long, sim_int_2, label='parameter1_2K', linestyle="None", marker='.')
ax.set_ylim(top=0.015, bottom=-0.0001)

# plt.legend()
plt.show()


field_z = np.linspace(0,8,401)
# field = np.array([[0, 0, bi] for bi in field_z])
# field=[0,0,4]
Mx_list = []
Mz_list = []
Mtot_list = []
for idx,b in enumerate(field_z):
    field = [b,0,0.6]
    # field = [0,0,b]
    M = HVS.magnetization(ion='Ho3+', Temp=2,Field=field)
    Mx_list.append(np.abs(M[0]))
    Mz_list.append(np.abs(M[2]))
    Mtot_list.append(np.sqrt(M[0]**2+M[1]**2+M[2]**2))
    print(M)
fig2,ax2 = plt.subplots()
ax2.plot(field_z,Mx_list,label='$M_x$',linestyle="None",marker='.')
ax2.plot(field_z,Mz_list,label='$M_z$',linestyle="None",marker='.')
ax2.plot(field_z,Mtot_list,label='$M_{tot}$',linestyle="None",marker='.')
plt.legend(loc='best')
plt.show()


# field_z = np.linspace(0,2,401)
#
# Mx_list = []
# Mz_list = []
# Mtot_list = []
# for idx,b in enumerate(field_z):
#     field = [0,0,b]
#     M = TVS.magnetization(ion='Tb3+', Temp=0.4,Field=field)
#     Mx_list.append(np.abs(M[0]))
#     Mz_list.append(np.abs(M[2]))
#     Mtot_list.append(np.sqrt(M[0]**2+M[1]**2+M[2]**2))
#     # print(M)
# fig2,ax2 = plt.subplots()
# ax2.plot(field_z,Mx_list,label='magnetization',linestyle="None",marker='.')
# ax2.plot(field_z,Mz_list,label='magnetization',linestyle="None",marker='.')
# ax2.plot(field_z,Mtot_list,label='magnetization',linestyle="None",marker='.')
# plt.show()