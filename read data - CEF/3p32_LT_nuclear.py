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

def spectrum(obj, Earray, Temp, ResFunc, gamma = 0):
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
        if eigenvector[1]!=0 and abs(eigenvector[7])<abs(eigenvector[1]):
            # print(idx)
            levels.append(idx)
            # print(obj.eigenvalues[idx])
        elif eigenvector[2]!=0 and abs(eigenvector[8])<abs(eigenvector[2]):
            # print(idx)
            levels.append(idx)
            # print(obj.eigenvalues[idx])
    excitation = obj.eigenvalues[levels[1]]-obj.eigenvalues[levels[0]]
    return excitation


# def add_field(obj,field):
#     muB = 0.05788
#     mJ = np.diag([6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6])
#     g = 3 / 2
#     obj.H += g * muB  * mJ * field
#     new_obj = obj.diagonalize()
#     return new_obj


ion = 'Tb3+'
folder = '../data/ASCII_file/'
ei_list = [3.32]
t_list = [10,20, 30]
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
e_range_h1 = [3, 10]  ## plot range
e_range_l = [0.2, 2.7]

# rat_list = [1.9662248916805452, 1.9881952766649167, 1.9991492029197095, 1.985101829103339]    ##Bragg peak (0 0 2)
# rat_list = [1.5678698062123513, 1.557786839117861, 1.5517928090765991, 1.5317630316596096]  ##incoherent
# lor_bg = [[8.803479, -0.670054, 0.001893, 0.000097],
#           [3.944630, -0.204455, 0.001277, 0.000314],
#           [4.078812, -0.169387, 0.001407, 0.000438],
#           [2.890807, -0.139814, 0.002854, 0.000790]]
lor_bg = [[4.590443, -0.710718, 0.004283, -0.000117],
          [3.587420, -0.094294, 0.000740, 0.000141],
          [0.749444, -0.043719, 0.003279, 0.000014],
          [1.614562, -0.032381, 0.002268, -0.000238]]
linear_bg = [[-0.000168, 0.001401],
             [-0.000180, 0.001606],
             [-0.000180, 0.001641],
             [-0.000180, 0.001647]]

# gamma_3 = [0.563855, 0.291993, 0.301208, 0.3817]
# gamma_12 = [0.322520, 0.2850965, 0.3430645, 0.403278]
gamma_3 = [0.563855, 0.291993, 0.301208, 0.3817]
gamma_12 = [0.322520, 0.331566, 0.3430645, 0.403278]

comb_e_list = []
comb_i_list = []
comb_di_list = []
int_3p32_list = []
int_12_list = []
un_12_list = []
un_3p32_list = []
for idx, measurement_3 in enumerate(data_list_3p32):
    ###--------------
    measurement_3 = data_list_3p32[idx]
    q = measurement_3.q
    e = measurement_3.e
    mask_i = measurement_3.mI
    mask_di = measurement_3.mdI

    # conax.pcolormesh(q, e, mask_i, cmap='jet', vmin=energeRange[0], vmax=energeRange[1])
    # conax.vlines(min(q_range), ymin=min(e_range_l), ymax=max(e_range_h), ls='--', color='r')
    # conax.vlines(max(q_range), ymin=min(e_range_l), ymax=max(e_range_h), ls='--', color='r')
    # conax.hlines(min(e_range_l), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    # conax.hlines(max(e_range_h), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')

    x_e_3, sumII_q_3, err_q_3 = disp(q, e, mask_i, mask_di, q_range, e_range_l)
    int_3p32_list.append(sumII_q_3)
    un_3p32_list.append(err_q_3)
fig,axes = plt.subplots()
for idx, t in enumerate(t_list):
    # FittedSpectrum = TVS.normalizedNeutronSpectrum(comb_e, Temp=t, ResFunc=ins_res) * FitCoefRes1['prefactor']
    # FittedSpectrum =spectrum(TVS,x_e_12, Temp=t,ResFunc=ins_res_12,gamma=gamma_12[idx]) * FitCoefRes1['prefactor']
    # FittedSpectrum2 = FitCoefRes1['ratio']*spectrum(TVS,x_e_3, Temp=t,ResFunc=ins_res_3p32,gamma=gamma_3[idx]) * FitCoefRes1['prefactor']
    # axes.errorbar(x_e_12,  int_12_list[idx],yerr=un_12_list[idx], marker='.', ls='none', label='{} K'.format(t), color="C{}".format(idx))
    axes.errorbar(x_e_3, int_3p32_list[idx],yerr=un_3p32_list[idx], marker='.', ls='none', color="C{}".format(idx),label='{} K'.format(t))
    # axes.errorbar(int_12_list[idx],x_e_12, xerr=un_12_list[idx], marker='.', ls='none', label='{} K'.format(t), color="C{}".format(idx))
    # axes.errorbar(int_3p32_list[idx],x_e_3, xerr=un_3p32_list[idx], marker='.', ls='none', color="C{}".format(idx))
axes.set_xlabel('$\\Delta E$ (meV)',fontsize=20)
axes.set_ylabel('Intensity (a.u.)',fontsize=20)
axes.tick_params(axis='both', which='major', labelsize=20)
axes.set_xlim(left=0,right=3)
axes.set_ylim(bottom=-0.000,top=0.006)

# plot labels
# plt.legend(loc='lower right',fontsize=20)
plt.legend(loc='best',fontsize=18,handletextpad=0.01,columnspacing=0.3)
plt.tight_layout()

# plt.savefig("../plots/comb_cut_setI",dpi=300)
plt.show()


