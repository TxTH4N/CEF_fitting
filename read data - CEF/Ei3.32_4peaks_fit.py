import numpy as np
import matplotlib.pyplot as plt
# import PyCrystalField as cef
from scipy.optimize import curve_fit
import scipy.special as spf
import scipy.constants as cts
from lmfit import models, Parameters


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
    y = amplitude * np.exp(-(x - mean) ** 2 / 2 / (sigma ** 2))
    return y


def exp_dec(x, a, beta, c):
    y = a * np.exp(-x * beta) + c
    return y


# def multi_gaussian(x, exp_amp,exp_beta,bg_const,a1,pos1,fw1,a2,pos2,fw2,a3,pos3,fw3,a4,pos4,fw4):
#     y = exp_dec(x, exp_amp, exp_beta, bg_const)
#     y += gaussian(x, a1, pos1, fw1)
#     y += gaussian(x, a2, pos2, fw2)
#     y += gaussian(x, a3, pos3, fw3)
#     y += gaussian(x, a4, pos4, fw4)
#     return y

def multi_gaussian(x, *params):
    a = params[0]
    beta = params[1]
    c = params[2]
    y = exp_dec(x, a, beta, c)
    # fwhm = params[3]
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
path = '../data/ASCII_file/'
Ei = 3.32
T = 1.7
file = 'Ei{}_T{}.iexy'.format('3p32', '1p7')
q, e, data, mask_i, mask_di = load_data(path, file)
##---------------

# energeRange = [0, 10]
energeRange = [0, 0.01]

# # ##------ the intensity is average of specific energy level
q_range = [0.6, 1.8]  ## q bin range
e_range = [0.29, 2.4]  ## plot range
# e_range = [1.2, 1.7]  ## plot range

config, conax = contourPlot(q, e, mask_i, energeRange)
conax.set_ylim(bottom=-0.1)
conax.set_xlim(right=2.3)

x_e, sumII_q, err_q = disp(q, e, mask_i, mask_di, q_range, e_range)

fig1, ax1 = plt.subplots()
ax1.errorbar(x_e, sumII_q, err_q, marker='.', ls='none', mfc='None', label='data')
ax1.set_xlabel('$\\hbar \\omega$ (meV)')
ax1.set_ylabel('$\\rm I$ (a.u.)')
ax1.set_title('Energy dispersion at T = {} K'.format(T))

fit_range = [0.35, 2.25]

# fmod = Model(multi_gaussian)
model = (models.ExponentialModel(prefix='exp_')
         + models.GaussianModel(prefix='g1_')
         + models.GaussianModel(prefix='g2_')
         + models.GaussianModel(prefix='g3_')
         + models.GaussianModel(prefix='g4_')
         )
params = model.make_params()
params['exp_amplitude'].set(0.0001)
params['exp_decay'].set(1)

# params['g1_center'].set(expr='1.306')
# params['g2_center'].set(expr='1.40')
# params['g3_center'].set(expr='1.55')
# params['g4_center'].set(expr='1.640')
params['g1_center'].set(1.3, min=1.15, max=1.5)
params['g1_amplitude'].set(0.00005, min=0)
params['g1_sigma'].set(0.01, min=0)


params['g2_center'].set(1.4, min=1.15, max=1.7)
params['g2_amplitude'].set(0.000005 , min=0)
params['g2_sigma'].set(expr='g1_sigma')

params['g3_center'].set(1.5, min=1.15, max=1.7)
params['g3_amplitude'].set(0.000005, min=0)
params['g3_sigma'].set(0.01, min=0,expr = 'g1_sigma')
#
params['g4_center'].set(1.6, min=1.2, max=1.7)
params['g4_amplitude'].set(0.00005, min=0)
params['g4_sigma'].set(expr='g1_sigma')

# params['g4_sigma'].set(expr='g3_sigma')
result = model.fit(sumII_q, params, x=x_e)
print(result.fit_report())
ax1.plot(x_e,result.best_fit,label='fit')
components = result.eval_components(x=x_e)
ax1.plot(x_e,components['g1_']+components['exp_'],label='gaussian1')
ax1.plot(x_e,components['g2_']+components['exp_'],label='gaussian2')
ax1.plot(x_e,components['g3_']+components['exp_'],label='gaussian3')
ax1.plot(x_e,components['g4_']+components['exp_'],label='gaussian4')

ax1.set_xlim(left=1,right =2)
plt.legend()
plt.show()
