import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
from scipy.optimize import curve_fit
import scipy.special as spf
import scipy.constants as cts
from lmfit import Model, Parameters
from lmfit import create_params, fit_report, minimize
import matplotlib
import load_data as ld
import timeit
from pcf_lib.Operators import Ket
import renormalize_ratio as rr


# matplotlib.use('QtAgg')

def lorentzian(x, amplitude, mean, gamma, const):
    y = amplitude / np.pi * 0.5 * gamma / ((x - mean) ** 2 + (0.5 * gamma) ** 2) + const
    return y


def linear(x, slope, intercept):
    y = slope * x + intercept
    return y


def contourPlot(q, e, int, int_ran):
    fig, ax = plt.subplots()
    cp = ax.pcolormesh(q, e, int, cmap='jet', vmin=int_ran[0], vmax=int_ran[1])
    fig.colorbar(mappable=cp)
    ax.set_xlabel("Q $(Å^{-1})$")
    ax.set_ylabel("$\\hbar \\omega$ (meV)")
    # ax.vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    # ax.vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    # ax.hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    # ax.hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    # fig.show()
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


def spectrum(obj, Earray, Temp, ResFunc, gamma=0):
    eigenkets = obj.eigenvectors.real
    intensity = np.zeros(len(Earray))

    beta = 1 / (8.61733e-2 * Temp)
    Z = sum([np.exp(-beta * en) for en in obj.eigenvalues])

    for i, ket_i in enumerate(eigenkets):
        pn = np.exp(-beta * obj.eigenvalues[i]) / Z
        if pn > 1e-3:  # only compute for transitions with enough weight
            for j, ket_j in enumerate(eigenkets):
                # compute amplitude
                # mJn = self._transition(ket_i,ket_j)  # Old: slow
                mJn = obj.opttran.transition(ket_i, ket_j)
                deltaE = obj.eigenvalues[j] - obj.eigenvalues[i]
                if deltaE > 0.2:
                    GausWidth = ResFunc(deltaE)  # peak width due to instrument resolution
                    intensity += ((pn * mJn * obj._voigt(x=Earray, x0=deltaE, alpha=GausWidth,
                                                         gamma=gamma)).real).astype('float64')

    return intensity


def ex_energy(obj):
    obj.diagonalize()
    levels = []
    for idx, eigenvector in enumerate(obj.eigenvectors):
        if eigenvector[1] != 0 and abs(eigenvector[7]) < abs(eigenvector[1]):
            # print(idx)
            levels.append(idx)
            # print(obj.eigenvalues[idx])
        elif eigenvector[2] != 0 and abs(eigenvector[8]) < abs(eigenvector[2]):
            # print(idx)
            levels.append(idx)
            # print(obj.eigenvalues[idx])
    excitation = obj.eigenvalues[levels[1]] - obj.eigenvalues[levels[0]]
    return excitation


def ins_res_3p32(x):
    y = 0.00030354 * x ** 3 + 0.0039009 * x ** 2 - 0.040862 * x + 0.11303
    return y


def ins_res_12(x):
    y = 3.8775e-05 * x ** 3 + 0.0018964 * x ** 2 - 0.078275 * x + 0.72305
    return y


####-------------------def functions above ----------------------

ion = 'Tb3+'
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

energeRange = [0, 0.01]
q_range = [1.3, 1.9]  ## q bin range
e_range_h = [3, 10]  ## plot range
e_range_l = [0.2, 3]

print(data_list_12[0].__dir__())
exp_bg = [[8.803479, -0.670054, 0.001893, 0.000097],
          [3.944630, -0.204455, 0.001277, 0.000314],
          [4.078812, -0.169387, 0.001407, 0.000438],
          [2.890807, -0.139814, 0.002854, 0.000790]]
comb_e_list = []
comb_i_list = []
comb_di_list = []
int_3p32_list = []
int_12_list = []
un_12_list = []
un_3p32_list = []
for idx, measurement_12 in enumerate(data_list_12):
    q = measurement_12.q
    e = measurement_12.e
    mask_i = measurement_12.mI
    mask_di = measurement_12.mdI

    # config, conax = plt.subplots()
    # cp = conax.pcolormesh(q, e, mask_i, cmap='jet', vmin=energeRange[0], vmax=energeRange[1])
    # config.colorbar(mappable=cp)
    # conax.set_xlabel("Q $(Å^{-1})$")
    # conax.set_ylabel("$\\hbar \\omega$ (meV)")
    # conax.hlines(3, xmin=(min(q)), xmax=(max(q)), ls='--', color='yellow')
    # conax.set_ylim(bottom=-0.5)
    # conax.set_xlim(right=4.5)

    x_e_12, sumII_q_12, err_q_12 = disp(q, e, mask_i, mask_di, q_range, e_range_h)
    slope = (sumII_q_12[-1] - sumII_q_12[0]) / (x_e_12[-1] - x_e_12[0])
    intercept = sumII_q_12[0] - x_e_12[0] * slope

    ###--------------
    measurement_3 = data_list_3p32[idx]
    q = measurement_3.q
    e = measurement_3.e
    mask_i = measurement_3.mI
    mask_di = measurement_3.mdI

    x_e_3, sumII_q_3, err_q_3 = disp(q, e, mask_i, mask_di, q_range, e_range_l)
    bg_3 = lorentzian(x_e_3, *exp_bg[idx])
    bg_12 = linear(x_e_12, slope, intercept)
    # bg_12 = linear(x_e_12, *linear_bg[idx])

    int_3p32_list.append((sumII_q_3 - bg_3))
    int_12_list.append(sumII_q_12 - bg_12)
    comb_e = np.append(x_e_3, x_e_12)
    comb_e_list.append(comb_e)
    comb_i = np.append((sumII_q_3 - bg_3), sumII_q_12 - bg_12)
    comb_i_list.append(comb_i)

    ### -----------uncertainty calculation------------
    ### -----12meV, linear background-----
    slope_un = np.sqrt(err_q_12[-1] ** 2 + err_q_12[0] ** 2) / (x_e_12[-1] - x_e_12[0])
    intercept_un = np.sqrt(err_q_12[0] ** 2 + (slope_un * x_e_12[0]) ** 2)
    bg12_un = np.sqrt((x_e_12 * slope_un) ** 2 + intercept_un ** 2)
    un_12 = np.sqrt(err_q_12 ** 2 + bg12_un ** 2)
    un_12_list.append(un_12)
    ### -----3.32meV, lorentzian background-----
    un_3p32_list.append(err_q_3 * np.sqrt(2))

gamma_3 = [0.563855,0.166451, 0.225229,0.283669]
gamma_12 = [0.214231, 0.143924, 0.2075815, 0.253425]

# ---------try with pycrystalfield-------
# ratio_list=np.linspace(1,3,21)
# B66_list = np.linspace(0,60e-6,61)
# ratio_list = np.linspace(1., 1.5, 6)
# B66_list = np.linspace(0e-6, 5e-6, 6)
# ratio_list = np.linspace(1, 1.4, 5)
# B66_list = np.linspace(20e-6, 30e-6, 11)
# ratio_list = np.linspace(1.3, 1.8, 6)
# B66_list = np.linspace(48e-6, 54e-6, 7)
# ratio_list = np.linspace(1.5, 3.1, 1)
muB = 0.05788
mJ = np.diag([6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6])
g = 3 / 2

ratio = 1.15
field_list = np.linspace(0,1.5,16)
B40_list = np.linspace(-0.0006, -5e-4, 11)
chi_list = []

for field_idx, field_guess in enumerate(field_list):
    chi_rat = []
    prefactor = 0.003
    B20 = -0.1
    B60 = 2e-06
    B66 = 3e-6
    for b40_idx, B40 in enumerate(B40_list):
        print(B40)
        Bdictionary = {'B20': B20, 'B40': B40,
                       'B60': B60, 'B66': B66}
        TVS = cef.CFLevels.Bdict(ion, Bdictionary)
        TVS.diagonalize()
        print('Eigenvalues of Hamiltonian based on provided B:\n{}'.format(TVS.eigenvalues))
        TVS.printEigenvectors()


        # fig1, ax1 = plt.subplots()
        # sim_int_40 = 0.003 * TVS.normalizedNeutronSpectrum(x_e_3, Temp=40, ResFunc=ins_res_3p32,gamma=gamma_3[-1])
        # sim_int_40_2 = 0.003 * TVS.normalizedNeutronSpectrum(x_e_12, Temp=40, ResFunc=ins_res_12,gamma=gamma_12[-1])
        #
        # ax1.plot(x_e_3, sim_int_40, label='parameter1_40K_initial guess', linestyle="None", marker='.')
        # ax1.plot(x_e_12, sim_int_40_2, label='parameter1_40K_initial guess', linestyle="None", marker='.')
        #
        # ax1.set_title("Initial guess")
        def err_global(CFLevelsObject, coeff, prefactor):
            coeff[1] = B40
            CFLevelsObject.newCoeff(coeff)
            CFLevelsObject.H += g * muB * mJ * field_guess
            CFLevelsObject.diagonalize()
            sigsq = 0
            for idx, t in enumerate(t_list):
                sigsq += np.sum((prefactor * spectrum(CFLevelsObject, x_e_12, Temp=t, ResFunc=ins_res_12,
                                                      gamma=gamma_12[idx]) - int_12_list[idx]) ** 2)
                sigsq += np.sum((prefactor * spectrum(CFLevelsObject, x_e_3, Temp=t, ResFunc=ins_res_3p32,
                                                      gamma=gamma_3[idx]) - 1 / ratio * int_3p32_list[idx]) ** 2)
            # excitation = ex_energy(CFLevelsObject)
            # sigsq += (excitation-4.896)**2
            return sigsq

        # ************************************************************

        print(TVS.B)
        ## Fit to neutron data
        FitCoefRes1 = TVS.fitdata(chisqfunc=err_global, fitargs=['prefactor', 'coeff'],
                                  prefactor=prefactor, coeff=TVS.B)
        print(FitCoefRes1)
        TVS.newCoeff(FitCoefRes1['coeff'])
        TVS.H += g * muB * field_guess * mJ
        TVS.diagonalize()  # Hamiltonian=field_H)
        # FittedSpectrum = TVS.normalizedNeutronSpectrum(x_e_3, Temp=30, ResFunc =ins_res)*FitCoefRes1['prefactor1']
        # FittedSpectrum2 = TVS.normalizedNeutronSpectrum(x_e_12, Temp=30, ResFunc =ins_res)*FitCoefRes1['prefactor1']*rat_list[1]

        ###### Plot result
        # plt.figure(figsize=(4,3))
        chisq = 0
        for idx, t in enumerate(t_list):
            # figs, axes = plt.subplots()

            FittedSpectrum = spectrum(TVS, x_e_12, Temp=t, ResFunc=ins_res_12, gamma=gamma_12[idx]) * FitCoefRes1[
                'prefactor']
            FittedSpectrum2 = spectrum(TVS, x_e_3, Temp=t, ResFunc=ins_res_3p32, gamma=gamma_3[idx]) * FitCoefRes1[
                'prefactor']
            # axes.errorbar(x_e_12, int_12_list[idx], yerr=un_12_list[idx], marker='.', ls='none', label='data',
            #               color="C0")
            # axes.errorbar(x_e_3, 1 / ratio * int_3p32_list[idx], yerr=ratio*un_3p32_list[idx], marker='.', ls='none',
            #               label='data', color="C0")
            # axes.set_title("Fitted Spectrum at T = {} K".format(t))

            xi = np.append(x_e_3, x_e_12)
            cal = np.append(FittedSpectrum2, FittedSpectrum)
            obs = np.append(1 / ratio * int_3p32_list[idx], int_12_list[idx])
            # axes.plot(xi, cal, color='C1')
            un = np.append(1 / ratio * un_3p32_list[idx], un_12_list[idx])

            un_No0 = np.array([value for value in un if value != 0])
            idx_0 = np.array([index for index, value in enumerate(un) if value == 0])
            obs_filtered = np.array([value for index, value in enumerate(obs) if index not in idx_0])
            cal_filtered = np.array([value for index, value in enumerate(cal) if index not in idx_0])
            # if idx==0:
            #     continue
            chisq_T = np.sum((cal_filtered - obs_filtered) ** 2 / un_No0 ** 2) / len(un_No0)
            print("chi square at T = {} K is {}".format(t_list[idx], chisq_T))
            chisq += np.sum((cal_filtered - obs_filtered) ** 2 / un_No0 ** 2)
            red_chisq = chisq / len(un_No0) / (idx + 1)
            print("accumulated chi square for all T is {}".format(red_chisq))
            # axes.set_xlabel('$\\hbar \\omega$ (meV)')
            # axes.set_ylabel('Intensity (a.u.)')
            # axes.set_ylim(bottom=-0.001)
        print('chisq for field{} and B40 {} is {}'.format(field_guess, B40, red_chisq))
        chi_rat.append(red_chisq)
    chi_list.append(chi_rat)
    # print(TVS.B)
    # TVS.printEigenvectors()
    # plot labels
    # plt.legend()
    # plt.show()
print(chi_list)
chi_list = np.array(chi_list)
config, conax = plt.subplots()
cp = conax.pcolormesh(B40_list, field_list, chi_list, cmap='jet')
config.colorbar(mappable=cp)
conax.set_xlabel('B40')
conax.set_ylabel('Molecular Field (T)')
plt.show()
