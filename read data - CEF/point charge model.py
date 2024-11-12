import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
from scipy.optimize import curve_fit
import scipy.special as spf
import load_data as ld
import timeit
import scipy.constants as cts
from lmfit import Model, Parameters
import matplotlib
matplotlib.use('QtAgg')


def contourPlot(obj, int_ran, q_range, e_range):
    fig, ax = plt.subplots()
    cp = ax.pcolormesh(obj.q, obj.e, obj.mI, cmap='jet', vmin=int_ran[0], vmax=int_ran[1])
    fig.colorbar(mappable=cp)
    ax.set_title("cplot_{}K_Ei{}".format(obj.temp, obj.ei))
    ax.set_xlabel("Q $(Å^{-1})$")
    ax.set_ylabel("$\\hbar \\omega$ (meV)")
    ax.vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    ax.vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    ax.hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    ax.hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    return fig, ax


def disp(q, e, ii, di, Q_range, E_range):
    q_ind = np.where(np.logical_and(q >= Q_range[0], q <= Q_range[1]))[0]  ## get q bin index
    e_ind = np.where(np.logical_and(e >= E_range[0], e <= E_range[1]))[0]
    ex = np.array([ei for ei in e[e_ind]])
    qx = np.array([qi for qi in q[q_ind]])
    IIMat = ii[e_ind, :][:, q_ind]
    errMat = di[e_ind, :][:, q_ind]
    sumII = np.nanmean(IIMat, axis=1)
    err_sq = np.nansum(np.square(errMat), axis=1)
    n = np.sum(1 - np.isnan(errMat), axis=1)
    err = np.sqrt(err_sq) / n
    return ex, sumII, err


def exp_dec(x, a, beta, c):
    y = a * np.exp(-x * beta) + c
    return y

def voigt(x, amplitude, mean,sigma,gamma):
    voi = amplitude * spf.voigt_profile(x - mean, sigma, gamma)
    return voi

# ####--------------------------------------------
folder = '../data/ASCII_file/'
ei_list = [12, 3.32]
t_list = [10, 30, 40, 60]
data_list_12 = []
data_list_3p32 = []

start = timeit.default_timer()

for temp in t_list:
    for ei in ei_list:
        if ei == 12:
            data_list_12.append(ld.data(folder, ei, temp))
        else:
            data_list_3p32.append(ld.data(folder, ei, temp))
stop = timeit.default_timer()
print('Inport data sets in {} seconds '.format(stop - start))  ### 0-4 3.32meV, 5-9 12meV

Int_Range = [0, 0.01]
# Int_Range = [0, 1]
Int_Range2 = [0, 0.05]

q_range_12 = [1.2, 2.4]  ## q bin range
e_range_12 = [3, 10]  ## plot range
# e_range_12 = [0, 10]  ## plot range
q_range_3p32 = [0.6, 1.8]  ## q bin range
e_range_3p32 = [0.29, 3]  ## plot range
# e_range_3p32 = [0, 3]  ## plot range

xe_ex = []
II_ex = []
err_ex = []

fig1, ax1 = plt.subplots()
for idx, data2 in enumerate(data_list_3p32):
    data2.mI_nor = data2.mI / 3.020
    data2.mdI_nor = data2.mdI / 3.020
    data2.x_e, data2.sumII_q, data2.err_q = disp(data2.q, data2.e, data2.mI_nor, data2.mdI_nor, q_range_3p32,
                                                 e_range_3p32)
    data = data_list_12[idx]
    data.x_e, data.sumII_q, data.err_q = disp(data.q, data.e, data.mI, data.mdI, q_range_12, e_range_12)
    temp_x = np.append(data2.x_e, data.x_e)
    temp_II = np.append(data2.sumII_q + data.sumII_q[0], data.sumII_q)
    temp_err = np.append(np.sqrt(data2.err_q ** 2 + data.err_q[0] ** 2), data.err_q)
    xe_ex.append(temp_x)
    II_ex.append(temp_II)
    err_ex.append(temp_err)
    ax1.errorbar(temp_x, temp_II, temp_err, marker='.', ls='-', fmt='o-', mfc='None', label="{}K".format(data2.temp))
    # ax1.errorbar(data2.x_e, data2.sumII_q+data_list_12[idx].sumII_q[0], data2.err_q, marker='.', ls='-', fmt='o-', mfc='None', label='data')
    ax1.set_xlabel('$\\hbar \\omega$ (meV)')
    ax1.set_ylabel('$\\rm I$ (a.u.)')
    ax1.set_title('Energy dispersion')
    ax1.legend()

def ins_res(x):
    y3p32 = +0.00030354 * x ** 3 + 0.0039009 * x ** 2 - 0.040862 * x + 0.11303
    y12p0 = 75e-05 * x ** 3 + 0.0018964 * x ** 2 - 0.078275 * x + 0.72305
    res1 = (1 - np.sign(np.sign(x - 3.32) + 1)) * y3p32
    res2 = np.sign(np.sign(x - 3.32) + 1) * y12p0
    y = res1 + res2
    return y


# ### ---------fit peaks ------------------
fit_range = [0.29,9]
num_of_peaks = 5
#### temperature list is [10, 30, 40, 60] and the variable name is xe_ex, II_ex, err_ex
# print(max(xe_ex[2]))
ind = np.where(np.logical_and(xe_ex[2] >= fit_range[0], xe_ex[2] <= fit_range[1]))[0]
x_fit = np.array([i for i in xe_ex[2][ind]])
y_fit = np.array([i for i in II_ex[2][ind]])
err_fit = np.array([i for i in err_ex[2][ind]])

fig_fit, ax_fit = plt.subplots()
ax_fit.errorbar(xe_ex[2],II_ex[2],err_ex[2])

fmod = Model(exp_dec, prefix='bg_')
for i in range(num_of_peaks):
    fmod += Model(voigt, prefix='v{}_'.format(i + 1))
params = fmod.make_params()
print('************************************')
print('Start fitting...')
print(f'Parameter names: {fmod.param_names}')

params['bg_a'].set(5.54e-3)
params['bg_beta'].set(0.8155)
params['bg_c'].set(0.0,min=0)

params['v1_amplitude'].set(0.002)
params['v1_mean'].set(0.75)
params['v1_sigma'].set(0.05)
params['v1_gamma'].set(0.05)

params['v2_amplitude'].set(0.002)
params['v2_mean'].set(1.3)
params['v2_sigma'].set(0.05)
params['v2_gamma'].set(0.05)

params['v3_amplitude'].set(0.002)
params['v3_mean'].set(1.7)
params['v3_sigma'].set(0.05)
params['v3_gamma'].set(0.05)

params['v4_amplitude'].set(0.002)
params['v4_mean'].set(5)
params['v4_sigma'].set(0.05)
params['v4_gamma'].set(0.05)

params['v5_amplitude'].set(0.005)
params['v5_mean'].set(8)
params['v5_sigma'].set(0.05)
params['v5_gamma'].set(0.05)

result = fmod.fit(y_fit, params, x=x_fit)
ax1.plot(x_fit, result.best_fit)
comps = result.eval_components(x=x_fit)
print('************************************')
print('Fitting is done!')

for name, par in result.params.items():
     print("  %s: value=%f +/- %f " % (name, par.value, par.stderr))
ax_fit.plot(x_fit,result.best_fit,label='fit to data')
# for i in range(num_of_peaks):
#     ax_fit.plot(x_fit, comps['g{}_'.format(i+1)], label=r'Gaussian #{}'.format(i+1))
ax_fit.set_xlim(left=0.29,right=10)
# print(result.params['bg_a'].value)

# ##-------------- Point Charge Model below-------------------------------
Tb166LigP, TbP = cef.importCIF('../TbV6Sn6_edit.cif', 'Tb1', MaxDistance=3.7)
print(Tb166LigP.__dir__())
print(Tb166LigP.LigandNames)
print(Tb166LigP.B)

# print(len(symequiv)-len(Tb166LigP.LigandNames))
symequiv = []
num = 0
# nm = np.round(Tb166LigP.LigandNames,7)
for idx, lig in enumerate(Tb166LigP.LigandNames):
    if idx == 0:
        symequiv.append(num)
    elif Tb166LigP.LigandNames[idx] == Tb166LigP.LigandNames[idx - 1]:
        symequiv.append(num)
    else:
        num += 1
        symequiv.append(num)
print(symequiv)

hamiltonian = Tb166LigP.PointChargeModel(symequiv, LigandCharge=[1,1,1], printB=True, IonCharge=3)
hamiltonian.diagonalize()
print(hamiltonian.B)

#
#
# ###-------------fit using B--------------------
ion = 'Tb3+'
# B20_cal = -0.102575
# B40_cal = -0.000590838
# B60_cal = 2.24021e-6
# B66_cal = 2.5e-6
B20_cal = -0.1
B40_cal = -5e-5
B60_cal = 1e-6
B66_cal = 0
Bdictionary_cal = {'B20': B20_cal, 'B40': B40_cal,
               'B60': B60_cal, 'B66': B66_cal}
TVS_cal = cef.CFLevels.Bdict(ion, Bdictionary_cal)
TVS_cal.diagonalize()
ObservedEnergies = TVS_cal.eigenvalues
print(ObservedEnergies)
xx = np.linspace(0.3, 10, 2000)
sim_int_40 = 0.0005 * TVS_cal.normalizedNeutronSpectrum(xx, Temp=40, ResFunc=ins_res)
fig, ax = plt.subplots()
ax.plot(xx, sim_int_40, label='calculated from B factor', marker='.', markersize=2)
#
#
def GlobalError(LigandsObject, LigandCharge, symequiv):
    newH = LigandsObject.PointChargeModel(symequiv, LigandCharge=LigandCharge, printB=False)
    newH.diagonalize()
    CalculatedEnergies = np.around(newH.eigenvalues.real, 6)
    print(CalculatedEnergies)
    erro = np.nansum((CalculatedEnergies - ObservedEnergies) ** 2)
    print(erro, end='\r')
    return erro


fitargs = ['LigandCharge']
Tb3fit, FitVals = Tb166LigP.FitChargesNeutrons(chisqfunc=GlobalError, fitargs=fitargs,
                                               LigandCharge=[-2, -2, -2], symequiv=symequiv)
# Tb3fit.printLaTexEigenvectors()
print(Tb3fit.B)
B20 = -0.1201341870710157
B40 = -0.0003239181669234637
B60 = -5.1626433954496875e-08
B66 = 2.973431250882485e-06

Bdictionary = {'B20': B20, 'B40': B40,
               'B60': B60, 'B66': B66}
#
TVS = cef.CFLevels.Bdict(ion, Bdictionary)
TVS.diagonalize()
#
xx = np.linspace(0.3, 10, 2000)
sim_int_40 = 0.0005 * TVS.normalizedNeutronSpectrum(xx, Temp=40, ResFunc=ins_res)
ax.plot(xx, sim_int_40, linestyle="None", marker='.',label='point charge model')
ax.legend()
plt.show()
# ### -------------- fitting data ---------------------
# def GlobalError(LigandsObject, LigandCharge, gamma, prefactor, symequiv):
#     newH = LigandsObject.PointChargeModel(symequiv, LigandCharge=LigandCharge, printB=False,IonCharge=3)
#     newH.diagonalize()
#     erro = np.sum(((prefactor*newH.normalizedNeutronSpectrum(Earray=x_fit, Temp = 40,
#                    ResFunc= ins_res,gamma=gamma)
#                     +exp_dec(x_fit,result.params['bg_a'].value,result.params['bg_beta'].value,result.params['bg_c'].value)-result.best_fit)/err_fit)**2)
#     return erro
# # ##------- Fit point charges to neutron data-------------
# fitargs = ['LigandCharge','prefactor']
# Tb166fit, FitVals = Tb166LigP.FitChargesNeutrons(chisqfunc = GlobalError,  fitargs = fitargs,
#                          LigandCharge = [1,1,1], gamma=0.12, prefactor=20, symequiv=symequiv)

# def chargetoE (x,prefactor,Sn1,Sn3,V):
#     newH = Tb166LigP.PointChargeModel(symequiv,LigandCharge=[Sn1,Sn3,V],printB=False,IonCharge=3)
#     newH.diagonalize()
#     yy = (prefactor * newH.normalizedNeutronSpectrum(Earray=x, Temp=40,ResFunc=ins_res)
#           +exp_dec(x,result.params['bg_a'].value,result.params['bg_beta'].value,result.params['bg_c'].value))
#     return yy
# p0=[0.005,-1,-1,-3]
# paraFC, param_cov = curve_fit(chargetoE, x_fit, y_fit)
# print(paraFC)