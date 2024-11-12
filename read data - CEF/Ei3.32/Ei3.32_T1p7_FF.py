import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
from scipy.optimize import curve_fit
import scipy.special as spf
import matplotlib
matplotlib.use('QtAgg')
import scipy.constants as cts
from lmfit import Model, Parameters


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
    fig, ax = plt.subplots(figsize = (4,3))
    cp = ax.pcolormesh(q, e, int, cmap='jet', vmin=int_ran[0], vmax=int_ran[1])
    fig.colorbar(mappable=cp)
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
def disp_E(q, e, ii, di, Q_range, E_range):
    q_ind = np.where(np.logical_and(q >= Q_range[0], q <= Q_range[1]))[0]  ## get q bin index
    e_ind = np.where(np.logical_and(e >= E_range[0], e <= E_range[1]))[0]
    # ex = np.array([ei for ei in e[e_ind]])
    qx = np.array([qi for qi in q[q_ind]])
    IIMat = ii[e_ind, :][:, q_ind]
    errMat = di[e_ind, :][:, q_ind]
    sumII = np.nansum(IIMat, axis=0)
    err_sq = np.nansum(np.square(errMat), axis=0)
    # n = np.sum(1 - np.isnan(errMat), axis=0)
    n=1
    err = np.sqrt(err_sq) / n
    return qx, sumII, err


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
T = 1.7
file = 'Ei{}_T{}.iexy'.format('3p32', '1p7')
q, e, data, mask_i, mask_di = load_data(path, file)
##---------------

# energeRange = [0, 10]
energeRange = [0, 0.01]

# # ##------ the intensity is average of specific energy level
q_range = [0.43, 2]  ## q bin range
# q_range = [0.5, 1.95]
e_range = [1.2, 1.4]  ## plot range
# e_range = [1.6, 1.7]  ## plot range

# contourPlot(q, e, mask_i, energeRange)
config, conax = contourPlot(q, e, mask_i, energeRange)
conax.set_ylim(bottom=-0.1)
conax.set_xlim(right=2.3)



x_q, sumII_e, err_e = disp_E(q, e, mask_i, mask_di, q_range, e_range)

fig1, ax1 = plt.subplots()
ax1.errorbar(x_q, sumII_e, err_e, marker='.', ls='none', fmt='o-', mfc='None', label='data')
ax1.set_xlabel('$\\hbar \\omega$ (meV)')
ax1.set_ylabel('$\\rm I$ (a.u.)')
ax1.set_title('Energy dispersion')
ax1.set_xlim(left=1,right=1.8)
ax1.set_ylim(bottom=0)
# ax1.set_xlim(left=-1,right=2)
#
# plt.legend()
plt.show()
