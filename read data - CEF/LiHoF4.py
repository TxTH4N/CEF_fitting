import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef

def ins_res(x, ei=3.32):
    if ei == 3.32:
        y = +0.00030354 * x ** 3 + 0.0039009 * x ** 2 - 0.040862 * x + 0.11303
    elif ei == 12:
        y = 75e-05 * x ** 3 + 0.0018964 * x ** 2 - 0.078275 * x + 0.72305
    return y

Ho = [-0.0579, 3.09e-4, 3.51e-3,0, 0.54e-6, 6.31e-5, 1.71e-5]

Bdictionary = {'B20': Ho[0], 'B40': Ho[1], 'B44':Ho[2],'B4-4':Ho[3],
               'B60': Ho[4], 'B64': Ho[5], 'B6-4':Ho[6]}
LHF = cef.CFLevels.Bdict('Ho3+', Bdictionary)
LHF.diagonalize()
print('Eigenvalues of Hamiltonian based on provided B:\n{}'.format(LHF.eigenvalues))
LHF.printLaTexEigenvectors()
LHF.printEigenvectors()

x_e_long = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
sim_int_2 = 0.0005 * LHF.normalizedNeutronSpectrum(x_e_long, Temp=4, ResFunc= lambda x : 0.1)
ax.plot(x_e_long, sim_int_2, label='parameter1_2K', linestyle="None", marker='.')
ax.set_ylim(top=0.015, bottom=-0.0001)

# plt.legend()
plt.show()

#
# ### ----------- calculate CEF levels under FM phase (adding molecular field) ----------------------
# muB = 0.05788
# mJ = np.diag([8,7,6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6,-7, -8])
# S = 2
# L = 6
# J = 8
# g = 1 +(J*(J+1)+S*(S+1)-L*(L+1))/(2*J*(J+1))
# print(g)
#
# M = LHF.magnetization(ion='Ho3+', Temp=200,Field=[0,0,1])
# print(M)
# ### ----------- calculate neutron spectrum ----------------------
# x_e_long = np.linspace(0, 20, 1000)
# fig, ax = plt.subplots()
# sim_int_2 = 0.0005 * LHF.normalizedNeutronSpectrum(x_e_long, Temp=0.1, ResFunc= lambda x : 0.1)
# ax.plot(x_e_long, sim_int_2, label='parameter1_2K', linestyle="None", marker='.')
# ax.set_ylim(top=0.015, bottom=-0.0001)
# # plt.legend()
plt.show()

# field_z = np.linspace(0,8,801)
# # field = np.array([[0, 0, bi] for bi in field_z])
# # field=[0,0,4]
# Mx_list = []
# Mz_list = []
# Mtot_list = []
# for idx,b in enumerate(field_z):
#     Bmf_ini =0.3
#     field = [b,0,Bmf_ini]
#     M = HVS.magnetization(ion='Ho3+', Temp=0.2,Field=field)
#     Mz = M[2]
#
#     # Mx_list.append(np.abs(M[0]))
#     # Mz_list.append(np.abs(M[2]))
#     # Mtot_list.append(np.sqrt(M[0]**2+M[1]**2+M[2]**2))
#     # print(M)
# fig2,ax2 = plt.subplots()
# ax2.plot(field_z,Mx_list,label='$M_x$',linestyle="None",marker='.')
# ax2.plot(field_z,Mz_list,label='$M_z$',linestyle="None",marker='.')
# ax2.plot(field_z,Mtot_list,label='$M_{tot}$',linestyle="None",marker='.')
# ax2.set_ylim(top=8, bottom=0)
# ax2.set_xlim(left=0, right=8)
# ax2.set_xlabel('$B_x$ (T)',fontsize=14)
# ax2.set_ylabel(r'$M (\mu_B)$',fontsize=14)
# ax2.tick_params(axis='both',labelsize=12)
# plt.legend(loc='best',fontsize=14)
# plt.show()
