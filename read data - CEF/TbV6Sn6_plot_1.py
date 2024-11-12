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
e_range_h1 = [3, 10]  ## plot range
e_range_l = [0.2, 2.7]

print(data_list_12[0].__dir__())
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

    x_e_12, sumII_q_12, err_q_12 = disp(q, e, mask_i, mask_di, q_range, e_range_h1)
    # slope = (sumII_q_12[-1] - sumII_q_12[0]) / (x_e_12[-1] - x_e_12[0])
    # intercept = sumII_q_12[0] - x_e_12[0] * slope

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
    bg_3 = lorentzian(x_e_3, *lor_bg[idx])
    # bg_12 = linear(x_e_12, slope, intercept)
    bg_12 = linear(x_e_12, *linear_bg[idx])

    int_3p32_list.append((sumII_q_3 - bg_3))
    int_12_list.append(sumII_q_12 - bg_12)
    # comb_e = np.append(x_e_3, x_e_12_temp)
    # comb_e_list.append(comb_e)

    # comb_i = np.append(sumII_q_3+np.abs(min(sumII_q_3)), sumII_q_12-bg_12)
    # comb_i_list.append(comb_i)

    # comb_di = np.append(err_q_3 / rat_list[idx], err_q_12)
    # comb_di_list.append(comb_di)
    # fig0,ax0 = plt.subplots()
    # ax0.errorbar(comb_e,comb_i,comb_di, marker='.', ls='none', fmt='o-', mfc='None', label='{}'.format(t_list[idx]))
    # plt.legend()
    ### -----------uncertainty calculation------------
    ### -----12meV, linear background-----
    # slope_un = np.sqrt(err_q_12[-1]**2+err_q_12[0]**2)/(x_e_12[-1] - x_e_12[0])
    # intercept_un = np.sqrt(err_q_12[0]**2+(slope_un*x_e_12[0])**2)
    # bg12_un = np.sqrt((x_e_12*slope_un)**2+intercept_un**2)
    # un_12 = np.sqrt(err_q_12**2+bg12_un**2)
    # un_12_list.append(un_12)
    un_12_list.append(err_q_12 * np.sqrt(2))
    ### -----3.32meV, lorentzian background-----
    un_3p32_list.append(err_q_3*np.sqrt(2))
# def fit_width(x):
#     # y = +0.00030354 * x ** 3 + 0.0039009 * x ** 2 - 0.040862 * x + 0.11303
#     if x < 0.5:
#         y = 0.0567
#     elif x < 1 and x >= 0.5:
#         y = 0.2553
#     elif x < 1.5 and x >= 1:
#         y = 0.211
#     elif x < 3 and x >= 1.5:
#         y = 0.2267
#     elif x < 6 and x >= 3:
#         y = 0.403
#     elif x >= 6:
#         y = 0.306
#     return y

def ins_res_3p32(x):
    y = 0.00030354 * x ** 3 + 0.0039009 * x ** 2 - 0.040862 * x + 0.11303
    return y

def ins_res_12(x):
    y =  3.8775e-05 * x**3 +0.0018964 * x**2 -0.078275 * x +0.72305
    return y

# gamma_3 = [0.281915, 0.1460, 0.150604, 0.1907]
# gamma_12 = [0.160905, 0.119245, 0.1716035, 0.1978]

# gamma_3 = [0.563855,0.166451, 0.225229,0.283669]
# gamma_12 = [0.214231, 0.143924, 0.2075815, 0.253425]

# ---------try with pycrystalfield-------

prefactor =0.003

# rat = 1.15
# field_guess=0.89
# B20 = -0.1036
# B40 = -0.00059
# B60 = 2.3233e-06
# B66= -2e-5

rat = 1.1
field_guess=0
# field_guess=1.
# B20 = -0.103
# B40 = -0.00059
# B60 = 2.27e-06
# B66= 3e-6


B20 =-0.114
B40 = -5.4e-04
B60 =2.27e-06
B66 =2.4e-5

# B20 =-0.13
# B40 = -5e-04
# B60 =2.2e-06
# B66 =2.4e-5


Bdictionary = {'B20': B20, 'B40': B40,
               'B60': B60, 'B66': B66}
TVS = cef.CFLevels.Bdict(ion, Bdictionary)

muB = 0.05788
mJ = np.diag([6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6])
g = 3 / 2
#     obj.H += g * muB  * mJ * field
def err_global(CFLevelsObject, coeff, prefactor,ratio):
    # coeff[-1]=B66
    CFLevelsObject.newCoeff(coeff)
    CFLevelsObject.H += g * muB  * mJ * (field_guess)
    CFLevelsObject.diagonalize()
    sigsq = 0
    for idx, t in enumerate(t_list):
        # CFLevelsObject.newCoeff(coeff)
        # CFLevelsObject.H += g * muB * mJ * (field_guess - idx * 0.1)
        # CFLevelsObject.diagonalize()
        sigsq += np.sum((prefactor * spectrum(CFLevelsObject,x_e_12,Temp=t,ResFunc=ins_res_12,gamma=gamma_12[idx]) - int_12_list[idx]) ** 2)
        sigsq += np.sum((prefactor * spectrum(CFLevelsObject, x_e_3, Temp=t, ResFunc=ins_res_3p32, gamma=gamma_3[idx]) -  1/ratio*int_3p32_list[idx]) ** 2)
    # excitation = ex_energy(CFLevelsObject)
    # sigsq += (excitation-4.896)**2
    return sigsq
# ************************************************************

print(TVS.B)
## Fit to neutron data
# FitCoefRes1 = TVS.fitdata(chisqfunc=err_global, fitargs=['prefactor','coeff','ratio'],
#                         prefactor=prefactor, coeff=TVS.B,ratio=rat)
FitCoefRes1={'prefactor': np.array([0.00336237]), 'coeff': np.array([-1.04030384e-01, -5.91118442e-04,  2.31365750e-06, -3.30687620e-06]),'ratio': np.array([1.201]), 'Chisq': 1.515122616300393}
# FitCoefRes1={'prefactor': np.array([0.00350270]), 'coeff': np.array([-0.10383753193619379, -0.0005926446761967422, 2.315727674737092e-06, 3.306853608959591e-06]),'ratio': np.array([1.18]), 'Chisq': 1.515122616300393}
# FitCoefRes1 = {'prefactor': np.array([0.00334849]), 'coeff': np.array([-1.15101643e-01, -5.57e-04,  2.31e-06,  2.25e-05]),  'ratio': np.array([1.36]), 'Chisq': 0.0004937771065236914}

### test below###
# FitCoefRes1={'prefactor': np.array([0.00336237]), 'coeff': np.array([-1.08e-01, -5.9e-04,  2.31365750e-06, -3.30687620e-06]),'ratio': np.array([1.201]), 'Chisq': 1.515122616300393}


print(FitCoefRes1)

# TVS.newCoeff(FitCoefRes1['coeff'])
# TVS.newCoeff([-1.03626951e-01, -5.94046786e-04,  2.32326048e-06, -2.10712431e-06])
TVS.diagonalize()  # Hamiltonian=field_H)


###### Plot result
# plt.figure(figsize=(4,3))
TVS.newCoeff(FitCoefRes1['coeff'])
TVS.H += g * muB * field_guess * mJ
TVS.diagonalize()  # Hamiltonian=field_H)
chisq = 0
# figs, axes = plt.subplots(figsize=(4.5,3*2))
# figs, axes = plt.subplots(figsize=(6,4))
figs, axes = plt.subplots(figsize=(6,3))
for idx, t in enumerate(t_list):
    # FittedSpectrum = TVS.normalizedNeutronSpectrum(comb_e, Temp=t, ResFunc=ins_res) * FitCoefRes1['prefactor']
    FittedSpectrum =spectrum(TVS,x_e_12, Temp=t,ResFunc=ins_res_12,gamma=gamma_12[idx]) * FitCoefRes1['prefactor']
    FittedSpectrum2 = FitCoefRes1['ratio']*spectrum(TVS,x_e_3, Temp=t,ResFunc=ins_res_3p32,gamma=0) * FitCoefRes1['prefactor']
    axes.errorbar(x_e_12,  int_12_list[idx],yerr=un_12_list[idx], marker='.', ls='none', label='{} K'.format(t), color="C{}".format(idx))
    axes.errorbar(x_e_3, int_3p32_list[idx],yerr=un_3p32_list[idx], marker='.', ls='none', color="C{}".format(idx))
    # axes.errorbar(int_12_list[idx],x_e_12, xerr=un_12_list[idx], marker='.', ls='none', label='{} K'.format(t), color="C{}".format(idx))
    # axes.errorbar(int_3p32_list[idx],x_e_3, xerr=un_3p32_list[idx], marker='.', ls='none', color="C{}".format(idx))

    xi=np.append(x_e_3,x_e_12)
    cal = np.append(FittedSpectrum2, FittedSpectrum)
    obs = np.append(int_3p32_list[idx],  int_12_list[idx])
    axes.plot(xi,cal,color="C{}".format(idx),ls='-',marker='none')
    # axes.plot(cal, xi, color="C{}".format(idx), ls='-', marker='none')

    un = np.append(un_3p32_list[idx],un_12_list[idx])

    un_No0 = np.array([value for value in un if value != 0])
    idx_0 = np.array([index for index, value in enumerate(un) if value == 0])
    obs_filtered = np.array([value for index, value in enumerate(obs) if index not in idx_0])
    cal_filtered = np.array([value for index, value in enumerate(cal) if index not in idx_0])
    # if idx==0:
    #     continue
    chisq_T= np.sum((cal_filtered-obs_filtered)**2 / un_No0**2)/len(un_No0)
    print("chi square at T = {} K is {}".format(t_list[idx],chisq_T))
    chisq += np.sum((cal_filtered-obs_filtered)**2 / un_No0**2)
    red_chisq = chisq/((len(un_No0))*(idx+1)-5)
    print("accumulated chi square for all T is {}".format(red_chisq))
axes.set_xlabel('$\\Delta E$ (meV)',fontsize=20)
axes.set_ylabel('Intensity (a.u.)',fontsize=20)
axes.tick_params(axis='both', which='major', labelsize=20)
axes.set_xlim(left=0,right=10)
axes.set_ylim(bottom=-0.000,top=0.035)
# axes.text(8.1,0.025,r'$\Gamma_6 \rightarrow \Gamma_5$',fontsize=18)
# axes.text(4,0.0065,r'$\Gamma_5 \rightarrow \Gamma_4$',fontsize=18)
# axes.text(1.3,0.0095,r'$\Gamma_4 \rightarrow \Gamma_3$',fontsize=18)
# axes.text(1.07,0.0085,r'$\Gamma_3 \rightarrow \Gamma_2$',fontsize=18)
# axes.text(2.,0.0055,r'$\Gamma_2 \rightarrow \Gamma_1$',fontsize=18)
# axes.text(-0.52,0.015,r'$\Gamma_1 \rightarrow \Gamma_0$',fontsize=18)
print(TVS.B)
TVS.printEigenvectors()
# plot labels
# plt.legend(loc='lower right',fontsize=20)
plt.legend(loc='best',fontsize=18,ncol=1,handletextpad=0.01,columnspacing=0.3)
plt.tight_layout()

# plt.savefig("../plots/comb_cut_setII",dpi=300)
plt.show()

TVS.printLaTexEigenvectors()
