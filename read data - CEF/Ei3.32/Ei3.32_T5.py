import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
from scipy.optimize import curve_fit
import scipy.special as spf
import scipy.constants as cts
from lmfit import Model, Parameters
import matplotlib
matplotlib.use('QtAgg')


# ####--------crystal field calculation---------------
def point_charge():
    Tb166LigP, TbP = cef.importCIF('../TbV6Sn6_edit.cif', 'Tb1', Zaxis=[0, 0, 1], MaxDistance=5.5)
    # TbP.printEigenvectors()
    TbP.diagonalize()
    print("Eigenvalues of the point charge model:")
    print(TbP.eigenvalues)
    CalculatedSpectrum = TbP.normalizedNeutronSpectrum(x_e_long, Temp=2, ResFunc=lambda x: 0.1)
    fig, ax = plt.subplots()
    ax.plot(x_e_long, CalculatedSpectrum, label='point charge model')
    return Tb166LigP, TbP


def cry_fie_cal(Bdictionary):
    ion = 'Tb3+'
    # B20 = -0.0
    # B40 = -0.0007
    # B60 = 0.0
    # B66 = -8e-6

    TVS = cef.CFLevels.Bdict(ion, Bdictionary)
    TVS.diagonalize()
    print('********************************************')
    print('Eigenvalues of Hamiltonian based on provided B:\n{}'.format(TVS.eigenvalues))
    TVS.printEigenvectors()

    # Bdict = {'B20': B20 * cef.LSThet[ion][0] / cef.Thet[ion][0],
    #          'B40': B40 * cef.LSThet[ion][1] / cef.Thet[ion][1],
    #          'B60': B66 * cef.LSThet[ion][2] / cef.Thet[ion][2],
    #          'B66': B66 * cef.LSThet[ion][2] / cef.Thet[ion][2]}
    # Tb2 = cef.LS_CFLevels.Bdict(Bdict, 3, 3, -100000.)
    # Tb2.diagonalize()
    # print('Eigenvalues of Hamiltonian based on provided B on LS:\n{}'.format(Tb2.eigenvalues))
    # print('********************************************')

    # xx = np.linspace(0, 3, 1000)
    # yy = 0.005*TVS.normalizedNeutronSpectrum(Earray=xx, Temp=1.7, ResFunc=ins_res)
    # fig, ax = plt.subplots()
    # ax.plot(xx,yy, '-')
    # ax.set_xlabel('$\hbar \omega$ (meV)')
    # ax.set_ylabel('Intensity (a.u.)')
    ###--------------------------------

    return TVS


def load_data(path, file):
    # file ='/Users/tianxionghan/research/CrystalFieldCal/data/ASCII_file/Ei3p32_T1p7.iexy'
    data = np.genfromtxt(path + file, skip_header=1, unpack=True)
    q = np.unique(data[2])
    e = np.unique(data[3])
    n = len(e)
    mdata = np.ma.masked_where(data[0] == -1e+20, data[0])
    mdI = np.ma.masked_where(data[1] == -1, data[1])
    mii = np.transpose(np.array([mdata[i:i + n] for i in range(0, len(mdata), n)]))
    mdI = np.transpose(np.array([mdI[i:i + n] for i in range(0, len(mdI), n)]))
    mii2 = mii.copy()
    mii2[mii < -1e5] = None
    mdI2 = mdI.copy()
    mdI2[mdI < 0] = None
    print('Read data in the shape of {} in e and {} in q'.format(len(e), len(q)))
    return q, e, data, mii2, mdI2


def contourPlot(q, e, int, int_ran):
    fig, ax = plt.subplots()
    cp = ax.pcolormesh(q, e, int, cmap='jet', vmin=int_ran[0], vmax=int_ran[1])
    fig.colorbar(mappable=cp)
    ax.set_title("cplot_{}K_Ei{}".format(T, Ei))
    ax.set_xlabel("Q $(Å^{-1})$")
    ax.set_ylabel("$\\hbar \\omega$ (meV)")
    ax.vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    ax.vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    ax.hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    ax.hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
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


def gaussian(x, amplitude, mean, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    y = amplitude / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mean) ** 2 / 2 / (sigma ** 2))
    return y


def voigt(x, amp, pos, sig, gam):
    voi = amp * spf.voigt_profile(x - pos, sig, gam)
    return voi


def exp_dec(x, a, beta, c):
    y = a * np.exp(-x * beta) + c
    return y


def multi_voi(x, *params):
    a = params[0]
    beta = params[1]
    c = params[2]
    y = exp_dec(x, a, beta, c)
    for i in range(3, len(params), 4):
        amplitude = params[i]
        pos = params[i + 1]
        sig = params[i + 2]
        gam = params[i + 3]
        y += voigt(x, amplitude, pos, sig, gam)
    return y


def multi_gaussian(x, *params):
    a = params[0]
    beta = params[1]
    c = params[2]
    y = exp_dec(x, a, beta, c)
    for i in range(3, len(params), 3):
        amplitude = params[i]
        mean = params[i + 1]
        fwhm = params[i + 2]
        y += gaussian(x, amplitude, mean, fwhm)
    return y


def ins_res(x, ei=3.32):
    if ei == 3.32:
        y = +0.00030354 * x ** 3 + 0.0039009 * x ** 2 - 0.040862 * x + 0.11303
    elif ei == 12:
        y = 75e-05 * x ** 3 + 0.0018964 * x ** 2 - 0.078275 * x + 0.72305
    return y


def fit_peak(x, y, err, p0, x_range=[-np.inf, np.inf], bounds=(0, np.inf), fitfun=multi_gaussian):
    ind = np.where(np.logical_and(x >= x_range[0], x <= x_range[1]))[0]
    x_fit = np.array([i for i in x[ind]])
    y_fit = np.array([i for i in y[ind]])
    err_fit = np.array([i for i in err[ind]])
    pop, pco = curve_fit(fitfun, x_fit, y_fit, p0=p0, bounds=bounds, sigma=err_fit, absolute_sigma=True)

    # pop, pco = curve_fit(fitfun, x_fit, y_fit, p0=p0, bounds=bounds)
    unc = np.sqrt(np.diag(pco))
    return x_fit, y_fit, err_fit, pop, unc


def add_uncertainty(numbers, uncertainties):
    result = []
    for i, number in enumerate(numbers):
        uncertainty = uncertainties[i]
        decimal_part = "{:.8f}".format(uncertainty - int(uncertainty))[2:]
        for j, digit in enumerate(decimal_part):
            if digit != '0':
                uncertainty_power = j + 1
                break
        else:
            uncertainty_power = 0
        if uncertainty_power != 0:
            number = round(number, uncertainty_power)
            uncertainty = round(uncertainty, uncertainty_power) * 10 ** uncertainty_power
        result.append("{}({})".format(number, int(round(uncertainty))))
    return result


def plot(x, y, err):
    fig, ax = plt.subplots()
    ax.errorbar(x, y, err, marker='.', ls='none', fmt='o-', mfc='None', label='data')
    ax.set_xlabel('$\\hbar \\omega$ (meV)')
    ax.set_ylabel('$\\rm I$ (a.u.)')
    ax.set_title('Energy dispersion')
    fig.show()
    # ax.set_ylim(bottom=0)
    # ax.set_xlim(left=0)
    return fig, ax


def fit_plot(x, y, para, err, fitfun=multi_gaussian):
    f, ax = plot(x, y, err)
    ax.plot(x, fitfun(x, *para), zorder=10, label='fit to data')
    for i in range(3, len(para), 3):
        hliney = 0.5 * para[i] + exp_dec(para[i], para[0], para[1], para[2])
        ax.hlines(hliney, xmin=para[i] - ins_res(para[i], 12) / 2, xmax=para[i] + ins_res(para[i], 12) / 2,
                  color='r',
                  label='instrument resolution limit')
    # hliney = 0.5 * pop[3] + exp_dec(pop[4], pop[0], pop[1])
    # hliney2 = 0.5 * pop[6] + exp_dec(pop[7], pop[0], pop[1])
    # ax.hlines(hliney, xmin=pop[4] - ins_res(pop[4], 3.32) / 2, xmax=pop[4] + ins_res(pop[4], 3.32) / 2, color='r',
    #           label='instrument resolution limit')
    # ax.hlines(hliney2, xmin=pop[7] - ins_res(pop[7], 3.32) / 2, xmax=pop[7] + ins_res(pop[7], 3.32) / 2, color='r')
    plt.legend()
    return 0


# ####--------------------------------------------
path = '../../data/ASCII_file/'
Ei = 3.32
T = 5
file = 'Ei{}_T{}.iexy'.format('3p32', T)
q, e, data, mask_i, mask_di = load_data(path, file)

##---------------

energeRange = [0, 0.01]

# # ##------ the intensity is average of specific energy level
# q_range = [0.6, 1.8]  ## q bin range
# e_range = [0.29, 2.4]  ## plot range
q_range = [1.3,1.8]
e_range = [0.29, 2.4]

# contourPlot(q, e, mask_i, energeRange)
config, conax = contourPlot(q, e, mask_i, energeRange)
conax.set_ylim(bottom=-0.1)
conax.set_xlim(right=2.3)

x_e, sumII_q, err_q = disp(q, e, mask_i, mask_di, q_range, e_range)

fig1, ax1 = plt.subplots()
ax1.errorbar(x_e, sumII_q, err_q, marker='.', ls='none', fmt='o-', mfc='None', label='data')
ax1.set_xlabel('$\\hbar \\omega$ (meV)')
ax1.set_ylabel('$\\rm I$ (a.u.)')
ax1.set_title('Energy dispersion')

# ax1.scatter(x_e,exp_dec(x_e,0.005,0.5,0.0))

# ##----------negative energy plot, log scale-------------
# e_range2 = [-10, 0]  ## plot range
# x_en, sumII_qn, err_qn = disp(q, e, mask_i, mask_di, q_range, e_range2)
# ax1.errorbar(x_en, sumII_qn, err_qn, marker='.', ls='none', fmt='o-', mfc='None', label='data negative E')
# ax1.errorbar(-x_en, sumII_qn/np.exp(x_en/(8.617*10**(-2)*T)), err_qn/np.exp(x_en/(8.617*10**(-2)*T)),
#              marker='.', ls='none', fmt='o-', mfc='None', label='data negative E')

# fig2, ax2 = plot(x_e, np.log10(sumII_q), 1/np.log(10)*(err_q/sumII_q))
#
#
# ##--------------------------------------------------
text = ''
fig, ax = plt.subplots()
fit_range = [0.35, 2.25]
# p0 = [0.004, 3, 0.001,
#       0.005, 0.7, 0.05, 0.05,
#       0.005, 0.82, 0.05, 0.05]
p0=[5.54e-3,0.8155,0.0,
    0.005, 1.3, 0.05,
    0.005, 1.68, 0.05,
    ]
# fmod = Model(multi_voi)
# p0 = Parameters()
# # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
# p0.add_many(('exp_amp', 0.004, True, False, False, False, False, False),
#             ('exp_beta', 1, True, False, False, False, False, False),
#             ('exp_beta', 0.01, True, False, False, False, False, False,),
#             ('bg_const', 0.01, True, False, False, False, False, False,),
#             ('a1', 0.005, True, False, False, False, False, False),
#             ('pos1', 0.7, True, False, False, False, False, False),
#             ('sig1', 0.05, True, False, False, False, False, False),
#             ('gam1', 0.05, True, False, False, False, False, False),
#             ('a2', 0.005, True, False, False, False, False, False),
#             ('pos2', 0.8, True, False, False, False, False, False),
#             ('sig2', 0.05, True, False, False, False, False, False),
#             ('gam2', 0.05, True, False, False, False, False, False),
#             ('a3', 0.005, True, False, False, False, False, False),
#             ('pos3', 1.25, True, False, False, False, False, False),
#             ('sig3', 0.005, True, False, False, False, False, False),
#             ('gam3', 0.005, True, False, False, False, False, False),
#             )
#
fit_x, fit_y, err_fit, pop, unc = fit_peak(x_e, sumII_q, err_q, p0, fit_range, fitfun=multi_gaussian)
print('fitting parameters:\n{}'.format(pop))
print('uncertainties:\n{}'.format(unc))
ax1.plot(fit_x, multi_gaussian(fit_x, *pop))
for i in range(3, len(pop), 3):
    ax1.plot(fit_x, gaussian(fit_x, pop[i], pop[i + 1], pop[i+2])+exp_dec(fit_x,pop[0],pop[1],pop[2]))
para = add_uncertainty(pop, unc)
ax1.text(0.2 * max(x_e), 0.8 * max(sumII_q),
            'Gaussian peak:\ncenter1: {}meV, FWHM1: {}\n'
            'center2: {}meV, FWHM2: {}\nInstrument resolution:{} and {}\n'.format(
                 para[4], para[5], para[7], para[8],
                 round(ins_res(pop[4], 3.32), 3), round(ins_res(pop[7], 3.32), 3)), fontsize=10)

#
for i in range(3, len(pop), 3):
    # #     ins_lim = ins_res(pop[i+1],Ei)
    # #     print('FWHM of Gaussian peak {} @ {} : {}'.format((i+1)/3,np.round(pop[i+1],4),np.round(pop[i+2],5)))
    # #     print('Instrument resolution: {}'.format(ins_lim))
    ax.plot(fit_x, gaussian(fit_x, pop[i], pop[i + 1], pop[i+2]))
ax.errorbar(x_e, sumII_q - exp_dec(x_e, *pop[0:3]), err_q, marker='.', ls='none', fmt='o-', mfc='None', label='data')
ax.plot(fit_x, fit_x * pop[0] + pop[1])
ax.plot(fit_x, multi_gaussian(fit_x, *pop) - exp_dec(fit_x, *pop[0:3]), zorder=10, label='fit to data')
ax.set_ylim(top=0.01, bottom=0)
#


# para = add_uncertainty(pop, unc)
# for i in range(3, len(pop), 3):
#     hliney = 0.5 * pop[i] + exp_dec(pop[i + 1], pop[0], pop[1], pop[2])
#     axi.hlines(hliney, xmin=pop[i + 1] - ins_res(pop[i + 1], 3.32) / 2, xmax=pop[i + 1] + ins_res(pop[i + 1], 3.32) / 2,
#                color='r',
#                label='instrument resolution')
#     text += 'Gaussian peak{}, c = {}, fwhm = {}\n'.format(int(i / 3), para[i + 1], para[i + 2])
# text += 'Instrument resolution:{}\n'.format(round(ins_res(0, 3.32), 3))
#
# ##-----
# axi.legend()
# axi.text(0.2 * max(x_e), 0.05 * max(sumII_q), text, fontsize=9)
# axi.set_ylim(bottom=0)
# # fig.show()
#
# ###****************************************************
# ###------------ fit neutron with parameters-------------
# ###****************************************************
# print("-----------point charge model calculation --------------")
x_e_long = np.linspace(0, 12, 1000)
# Tb166LigP, TbP = point_charge()
#
#
# B20P = 0.3392873
# B40P = 0.00085596
# B60P = 2e-08
# B66P = 4.07e-06
# #
# BdictionaryP = {'B20': B20P, 'B40': B40P,
#                'B60': B60P, 'B66': B66P}
# TVS2 = cry_fie_cal(BdictionaryP)
# #
# print("-----------B factor  calculation --------------")
###
# B20 = 0.3392873
# B40 = 0.001
# B60 = 0
# B66 = 0
#
# B20 = 3.57790713e-01
# B40 = 1.00001845e-03
# B60 = 8.09445241e-09
# B66 = 5.64379638e-06

# ## parameters from simple hand calculation scenario 1
# B20 = -0.164674
# B40 = -0.000252118
# B60 = 0
# B66 = 0

## adding B60
B20 = -0.101908
B40 = -0.000594474
B60 = 2.26426e-6
B66 = 0

Bdictionary = {'B20': B20, 'B40': B40,
               'B60': B60, 'B66': B66}
#
TVS = cry_fie_cal(Bdictionary)
fig, ax = plt.subplots()
sim_int_40 = 0.0005 * TVS.normalizedNeutronSpectrum(x_e_long, Temp=40, ResFunc=ins_res)
ax.plot(x_e_long, sim_int_40, label='parameter1_40K', linestyle="None", marker='.')
ax.set_ylim(top=0.015, bottom=-0.0001)

# # ### fitting the neutron intensities
# ObservedEnergies = [pop[i+1] for i in range(3, len(pop), 3)]
# print(ObservedEnergies)
# def err_global(CFLevelsObject, coeff, prefactor):
#     """Global error function used for fitting"""
#
#     # define new Hamiltonian
#     CFLevelsObject.newCoeff(coeff)
#     erro = 0
#     # Compute error in neutron spectrum
#     calculatedNeutronSpectrum = prefactor * CFLevelsObject.normalizedNeutronSpectrum(Earray=x_e, Temp=40,
#                                                                                      ResFunc=ins_res)
#     erro += np.sum(((calculatedNeutronSpectrum - sumII_q-exp_dec(x_e,*pop[0:3])) / err_q) ** 2)
#
#     print(erro, end='\r')
#     return erro
# #
#
# # # ************************************************************
# # # Fit to neutron data
# # #
# gammaguess = [pop[i+2] for i in range(3, len(pop), 3)]
# # #
# FitCoefRes1 = TVS.fitdata(chisqfunc=err_global, fitargs=['coeff', 'prefactor'],
#                          coeff=TVS.B, prefactor=0.0005, method='Powell')  # fit from fitted PC
# #
# # TVS.newCoeff(FitCoefRes1['coeff'])
# FittedSpectrum = TVS.normalizedNeutronSpectrum(x_e, Temp=40, ResFunc = ins_res)*0.0005
# fig,ax = plt.subplots()
# ax.plot(x_e, FittedSpectrum, label='PyCrystalField fit')
# #

plt.legend()
plt.show()
