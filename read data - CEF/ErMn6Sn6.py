import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
import matplotlib
matplotlib.use('QtAgg')


def ins_res_3p32(x):
    y = 0.00030354 * x ** 3 + 0.0039009 * x ** 2 - 0.040862 * x + 0.11303
    return y

def ins_res_12(x):
    y =  3.8775e-05 * x**3 +0.0018964 * x**2 -0.078275 * x +0.72305
    return y


ion = 'Er3+'

## Ryan's parameters
B20 = 0.012
B40 = -3.69e-4
B60 = 0
B66 = 1.47e-5

## Aashish's parameters
# B20 = -0.022968
# B40 = 0.00012719
# B60 = 2.1198e-6
# B66 = 2.1107e-5

Bdictionary = {'B20': B20, 'B40': B40,
               'B60': B60, 'B66': B66}
print(Bdictionary)
#
HVS = cef.CFLevels.Bdict(ion, Bdictionary)
HVS.diagonalize()
print('Eigenvalues of Hamiltonian based on provided B:\n{}'.format(HVS.eigenvalues))
HVS.printEigenvectors()
# HVS.printLaTexEigenvectors()

temp_list = np.linspace(10,200,10)
neutron_spectrum_list = []
x_e_12 = np.linspace(3, 20, 1000)
x_e_3 = np.linspace(0, 3, 1000)
x_e = np.append(x_e_3, x_e_12)
Jx = HVS.opttran.Jx
Jy = HVS.opttran.Jy * 1j
Jz = HVS.opttran.Jz
muB = 5.7883818012e-2
kb = 0.08617
gJ = 5/4
S =2
L =6
bmf = 120/1000*kb*6*S/gJ/muB
print("Molecular field estimated:{} ".format(bmf))


def cal_molecular_field(cefobj,temperature,field):

    M = cefobj.magnetization(ion=ion, Temp=temperature,Field=field)
    return 0


for idx,temp_i in enumerate(temp_list):
    HVS = cef.CFLevels.Bdict(ion, Bdictionary)
    HVS.diagonalize()

# x_e_0 = np.linspace(0,20,1000)
    sim_int_1 = HVS.normalizedNeutronSpectrum(x_e_3, Temp=temp_i, ResFunc= ins_res_12)
    sim_int_2 = HVS.normalizedNeutronSpectrum(x_e_12, Temp=temp_i, ResFunc= ins_res_12)
    if idx == 0:
        fig, ax = plt.subplots()
        ax.plot(x_e_3, sim_int_1, label='parameter1_2K', linestyle="None", marker='.',color="C0")
        ax.plot(x_e_12, sim_int_2, label='parameter1_2K', linestyle="None", marker='.',color="C0")
        HVS.printLaTexEigenvectors()
        # plt.show()
    sim_int = np.append(sim_int_1, sim_int_2)
    neutron_spectrum_list.append(sim_int)
sim_int=np.array(neutron_spectrum_list)
rotated_intensity = sim_int.T
config, conax = plt.subplots()
energyRange = [0,20]
# cp = conax.pcolormesh( x_e,field_list, sim_int, cmap='jet',vmin=energyRange[0], vmax=energyRange[1])
cp = conax.pcolormesh( temp_list,x_e, rotated_intensity, cmap='jet',vmin=energyRange[0], vmax=energyRange[1])
config.colorbar(mappable=cp)
# conax.set_xlabel("Q $(Å^{-1})$")
# conax.set_ylabel("$\\hbar \\omega$ (meV)")

plt.show()