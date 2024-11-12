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
# rat_list = [1.9662248916805452, 1.9881952766649167, 1.9991492029197095, 1.985101829103339]    ##Bragg peak (0 0 2)
# rat_list = [1.5678698062123513, 1.557786839117861, 1.5517928090765991, 1.5317630316596096]  ##incoherent
exp_bg = [[8.803479, -0.670054, 0.001893, 0.000097],
          [3.944630, -0.204455, 0.001277, 0.000314],
          [4.078812, -0.169387, 0.001407, 0.000438],
          [2.890807, -0.139814, 0.002854, 0.000790]]
# linear_bg = [[-0.000443, 0.002574],
#              [-0.000223, 0.001606],
#              [-0.000353, 0.002188],
#              [-0.000208, 0.001787]]
# linear_bg = [[0, 0.000270],
#              [0, 0.000182],
#              [0, 0.000094],
#              [0, 0.000280]]

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

    # conax.pcolormesh(q, e, mask_i, cmap='jet', vmin=energeRange[0], vmax=energeRange[1])
    # conax.vlines(min(q_range), ymin=min(e_range_l), ymax=max(e_range_h), ls='--', color='r')
    # conax.vlines(max(q_range), ymin=min(e_range_l), ymax=max(e_range_h), ls='--', color='r')
    # conax.hlines(min(e_range_l), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    # conax.hlines(max(e_range_h), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')

    x_e_3, sumII_q_3, err_q_3 = disp(q, e, mask_i, mask_di, q_range, e_range_l)
    bg_3 = lorentzian(x_e_3, *exp_bg[idx])
    bg_12 = linear(x_e_12, slope, intercept)
    # bg_12 = linear(x_e_12, *linear_bg[idx])

    int_3p32_list.append((sumII_q_3 - bg_3))
    int_12_list.append(sumII_q_12 - bg_12)
    comb_e = np.append(x_e_3, x_e_12)
    comb_e_list.append(comb_e)
    comb_i = np.append((sumII_q_3 - bg_3),sumII_q_12 - bg_12)
    # comb_i = np.append(sumII_q_3+np.abs(min(sumII_q_3)), sumII_q_12-bg_12)
    comb_i_list.append(comb_i)

    # comb_di = np.append(err_q_3 / rat_list[idx], err_q_12)
    # comb_di_list.append(comb_di)
    # fig0,ax0 = plt.subplots()
    # ax0.errorbar(comb_e,comb_i,comb_di, marker='.', ls='none', fmt='o-', mfc='None', label='{}'.format(t_list[idx]))
    # plt.legend()
    ### -----------uncertainty calculation------------
    ### -----12meV, linear background-----
    slope_un = np.sqrt(err_q_12[-1]**2+err_q_12[0]**2)/(x_e_12[-1] - x_e_12[0])
    intercept_un = np.sqrt(err_q_12[0]**2+(slope_un*x_e_12[0])**2)
    bg12_un = np.sqrt((x_e_12*slope_un)**2+intercept_un**2)
    un_12 = np.sqrt(err_q_12**2+bg12_un**2)
    un_12_list.append(un_12)
    ### -----3.32meV, lorentzian background-----
    un_3p32_list.append(err_q_3*np.sqrt(2))
def ins_res(x):
    # y = +0.00030354 * x ** 3 + 0.0039009 * x ** 2 - 0.040862 * x + 0.11303
    if x < 0.5:
        y = 0.0567
    if x < 3 and x > 0.5:
        y = 0.4
    elif x < 6 and x >= 3:
        y = 0.64
    elif x >= 6:
        y = 0.531
    return y


#
#
# def gaussian(x, area, mean, fwhm):
#     sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
#     y = area / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mean) ** 2 / 2 / (sigma ** 2))
#     return y
#
#
# def multi_gaussian(x, *params):
#     c = params[0]
#     y = c
#     for i in range(1, len(params), 3):
#         area = params[i]
#         mean = params[i + 1]
#         fwhm = params[i + 2]
#         y += gaussian(x, area, mean, fwhm)
#     return y


# plt.tight_layout()
#
# ## try with lmfit
# def err_global(pars,xa,xb):
#     vals = pars.valuesdict()
#     pre = vals['prefactor']
#     rat = vals['ratio']
#     B20 = vals['B20']
#     B40 = vals['B40']
#     B60 = vals['B60']
#     B66 = vals['B66']
#     ion = 'Tb3+'
#     Bdictionary = {'B20': B20, 'B40': B40,
#                    'B60': B60, 'B66': B66}
#     TVS = cef.CFLevels.Bdict(ion, Bdictionary)
#     TVS.diagonalize()
#     resi = 0
#     for idx, tt in enumerate(t_list):
#         resi_12 = pre*TVS.normalizedNeutronSpectrum(Earray=xa, Temp=tt, ResFunc=ins_res)- int_12_list[idx]
#         resi_3= pre*rat* TVS.normalizedNeutronSpectrum(Earray=xb, Temp=tt, ResFunc=ins_res) -int_3p32_list[idx]
#         resi += np.append(resi_3,resi_12)
#     #   resi = np.sum(resi**2)
#     #     resi += np.sum((pre * TVS.normalizedNeutronSpectrum(Earray=xa, Temp=tt, ResFunc=ins_res) -int_12_list[idx])**2)
#     #     resi += np.sum((pre *rat *  TVS.normalizedNeutronSpectrum(Earray=xb, Temp=tt,ResFunc=ins_res) - int_3p32_list[idx])**2)
#     return resi
#
# # fit_params = create_params(prefactor =0.003,ratio = 0.5,B20 = -0.102,B40 = -0.000595,B60 = 2.26e-6,B66=2.5e-6)
# fit_params = create_params(prefactor =0.003,ratio = 2,B20 = -0.1,B40 = -0.0005,B60 = 2.e-6,B66=0)
#
#
# out = minimize(err_global,fit_params,args=(x_e_12,x_e_3))
# # out = minimize(err_global, fit_params, args=(comb_e_list,))
# print(fit_report(out))
#
# para_list = []
# for name, param in out.params.items():
#     para_list.append(param.value)
# Bdictionary = {'B20': para_list[2], 'B40': para_list[3],
#                'B60': para_list[4], 'B66': para_list[5]}
# TVS2 = cef.CFLevels.Bdict(ion, Bdictionary)
# TVS2.diagonalize()
# print('Eigenvalues of Hamiltonian based on provided B:\n{}'.format(TVS2.eigenvalues))
# TVS2.printEigenvectors()
# # plt.close("all")
#
# for idx, tem in enumerate(t_list):
#     para_list[1] * int_3p32_list[idx]
#
# for idx, tem in enumerate(t_list):
#     fig2, ax2 = plt.subplots()
#     # x_e_long = np.linspace(0.25, 12.2, 1000)
#     fit_reu = para_list[0] * TVS2.normalizedNeutronSpectrum(x_e_12, Temp=tem, ResFunc=ins_res)
#     fit_reu1 = para_list[0] * para_list[1] *TVS2.normalizedNeutronSpectrum(x_e_3, Temp=tem, ResFunc=ins_res)
#     ax2.plot(x_e_12, fit_reu, marker='.', ls='-', label="Fitted neutron spectrum")
#     ax2.plot(x_e_3, fit_reu1, marker='.', ls='-', label="Fitted neutron spectrum")
#     ax2.plot(x_e_12, int_12_list[idx], label='data', marker='.', ls='None')
#     ax2.plot(x_e_3,  int_3p32_list[idx], label='data', marker='.', ls='None')
#     ax2.set_ylim(bottom=0)
#     ax2.set_xlim(left=0)
#     ax2.set_xlabel(r'$\Delta E (meV)$', fontsize=8)
#     ax2.set_ylabel('Intensity (a. u.)', fontsize=8)
#     ax2.tick_params(axis='both', which='major', labelsize=8)
#     plt.legend(loc='best')
# print(TVS2.B)
# # TVS2.printLaTexEigenvectors()
# plt.tight_layout()
# plt.show()
# resi =np.sum((fit_reu-int_12_list[-1])**2)+np.sum((fit_reu1-int_3p32_list[-1]**2))
# print(resi)
#
#

# ---------try with pycrystalfield-------
# prefactor =0.003
# ratio = 2
# B20 = -0.1
# B40 = -0.0005
# B60 = 2.e-06
# B66= 1e-5

# prefactor =0.003
# ratio = 2
# B20 = -0.12
# B40 = -0.0006
# B60 = 2e-06
# B66= 3e-5

prefactor =0.003
# ratio = 2
ratio_list = [1.0706542995743615, 1.063883289821871, 1.0275472030236736, 1.0409803818157184]
# ratio = np.average(ratio_list)
ratio = 1.5
B20 = -0.1
B40 = -0.0006
B60 = 2e-06
B66= 24e-6

Bdictionary = {'B20': B20, 'B40': B40,
               'B60': B60, 'B66': B66}
TVS = cef.CFLevels.Bdict(ion, Bdictionary)
TVS.diagonalize()
print('Eigenvalues of Hamiltonian based on provided B:\n{}'.format(TVS.eigenvalues))
TVS.printEigenvectors()
# x_e_long = np.linspace(0.2, 12.2, 1000)
fig1, ax1 = plt.subplots()
sim_int_40 = 0.002 * TVS.normalizedNeutronSpectrum(comb_e, Temp=40, ResFunc=ins_res)
ax1.plot(comb_e, sim_int_40, label='parameter1_40K_initial guess', linestyle="None", marker='.')
# ax1.plot(comb_e,comb_i_list[2])
ax1.set_title("Initial guess")

def err_global(CFLevelsObject, coeff, prefactor,ratio):
    # coeff[-1]=B66
    CFLevelsObject.newCoeff(coeff)
    sigsq = 0
    for idx, t in enumerate(t_list):
        sigsq += np.sum((prefactor * CFLevelsObject.normalizedNeutronSpectrum(Earray=x_e_12, Temp=t, ResFunc=ins_res) - int_12_list[idx]) ** 2)
        sigsq += np.sum((prefactor*ratio* CFLevelsObject.normalizedNeutronSpectrum(Earray=x_e_3, Temp=t,ResFunc=ins_res) -int_3p32_list[idx]) ** 2)
    return sigsq


# ************************************************************

print(TVS.B)
## Fit to neutron data
FitCoefRes1 = TVS.fitdata(chisqfunc=err_global, fitargs=['prefactor','coeff','ratio'],
                        prefactor=prefactor, coeff=TVS.B,ratio=ratio)
print(TVS.B)
print(FitCoefRes1)
TVS.newCoeff(FitCoefRes1['coeff'])
print(TVS.B[3])
# FittedSpectrum = TVS.normalizedNeutronSpectrum(x_e_3, Temp=30, ResFunc =ins_res)*FitCoefRes1['prefactor1']
# FittedSpectrum2 = TVS.normalizedNeutronSpectrum(x_e_12, Temp=30, ResFunc =ins_res)*FitCoefRes1['prefactor1']*rat_list[1]

###### Plot result
# plt.figure(figsize=(4,3))
chisq = 0
for idx, t in enumerate(t_list):
    figs, axes = plt.subplots()
    # FittedSpectrum = TVS.normalizedNeutronSpectrum(comb_e, Temp=t, ResFunc=ins_res) * FitCoefRes1['prefactor']
    FittedSpectrum = TVS.normalizedNeutronSpectrum(x_e_12, Temp=t, ResFunc=ins_res) * FitCoefRes1['prefactor']
    FittedSpectrum2 = FitCoefRes1['ratio']* TVS.normalizedNeutronSpectrum(x_e_3, Temp=t, ResFunc=ins_res) * FitCoefRes1['prefactor']
    axes.errorbar(x_e_12, int_12_list[idx],yerr=un_12_list[idx], marker='.', ls='none', label='data', color="C0")
    axes.errorbar(x_e_3, int_3p32_list[idx],yerr=un_3p32_list[idx], marker='.', ls='none', label='data', color="C0")
    axes.set_title("Fitted Spectrum at T = {} K".format(t))
    # axes.plot(x_e_12, FittedSpectrum, label='fitted model', color="C1")
    # axes.plot(x_e_3, FittedSpectrum2, label='fitted model', color="C1")

    xi=np.append(x_e_3,x_e_12)
    cal = np.append(FittedSpectrum2, FittedSpectrum)
    obs = np.append(int_3p32_list[idx], int_12_list[idx])
    axes.plot(xi,cal,color='C1')
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
    red_chisq = chisq/len(un_No0)/(idx+1)
    print("accumulated chi square for all T is {}".format(red_chisq))
print(TVS.B)
# plot labels
plt.legend()
plt.xlabel('$\\hbar \\omega$ (meV)')
plt.ylabel('Intensity (a.u.)')
plt.tight_layout()
plt.show()

# filename= '/Users/tianxionghan/research/CEF_cal/CrystalFieldCal/plots/60K_scaled intensities.txt'
# with open(filename,'w') as file:
#     file.write("energy(meV)\tIntensities(Calculated)\tIntensities(Observed)\tuncertainty\n")
#     for idx, x_e in enumerate(x_e_3):
#         file.write(f"{x_e_3[idx]}\t{FittedSpectrum2[idx]}\t{1/FitCoefRes1['rat'][0]*int_3p32_list[-1][idx]}\t{1/FitCoefRes1['rat'][0]*un_3p32_list[-1][idx]}\n")
#     for idx, x_e in enumerate(x_e_12):
#         file.write(f"{x_e_12[idx]}\t{FittedSpectrum[idx]}\t{int_12_list[-1][idx]}\t{un_12_list[-1][idx]}\n")
