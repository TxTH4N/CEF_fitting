import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')
import PyCrystalField as cef
from scipy.optimize import curve_fit
import scipy.special as spf
import scipy.constants as cts
from lmfit import Model, Parameters
from lmfit.models import SkewedGaussianModel, LinearModel

# ratio (Ei12/Ei3p32) = 1.0004336710896622
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
    # ax.set_title("cplot_{}K_Ei{}".format(T, Ei))
    # ax.set_xlabel("Q $(Å^{-1})$")
    ax.set_ylabel("$\\hbar \\omega$ (meV)",fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
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

def dispQ(q, e, ii, di, Q_range, E_range):
    q_ind = np.where(np.logical_and(q >= Q_range[0], q <= Q_range[1]))[0]  ## get q bin index
    e_ind = np.where(np.logical_and(e >= E_range[0], e <= E_range[1]))[0]
    ex = np.array([ei for ei in e[e_ind]])
    qx = np.array([qi for qi in q[q_ind]])
    IIMat = ii[e_ind, :][:, q_ind]
    errMat = di[e_ind, :][:, q_ind]
    # sumII = np.nanmean(IIMat, axis=0)
    # err_sq = np.nansum(np.square(errMat), axis=0)
    # n = np.sum(1 - np.isnan(errMat), axis=0)
    # err = np.sqrt(err_sq) / n
    sumII = np.nansum(IIMat, axis=0)
    err_sq = np.nansum(np.square(errMat), axis=0)
    err = np.sqrt(err_sq)
    return qx, sumII, err


# def gaussian(x, amplitude, mean, fwhm):
#     sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
#     y = amplitude * np.exp(-(x - mean) ** 2 / 2 / (sigma ** 2))
#     return y
def gaussian(x, amplitude, mean, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    y = amplitude/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x - mean) ** 2 / 2 / (sigma ** 2))
    return y

def lorentzian(x, amplitude, mean, gamma, const):
    y = amplitude / np.pi * 0.5 * gamma / ((x - mean) ** 2 + (0.5 * gamma) ** 2) + const
    return y

def linear(x, slope, intercept):
    y = slope * x + intercept
    return y

def voigt(x, amp, pos, sig, gam):
    voi = amp*spf.voigt_profile(x - pos, sig, gam)
    return voi

def const(x,a):
    y =0*x+a

def exp_dec(x, a, beta, c):
    y = a * np.exp(-x * beta) + c
    return y


def multi_voi(x, *params):
    # a = params[0]
    # beta = params[1]
    # c = params[2]
    # y = exp_dec(x, a, beta, c)
    a = params[0]
    b = params[1]
    y = a*x+b
    fac = (1 - np.sign(np.sign(x) + 1))
    # det_bac = np.exp(x / (0.08617 * T) * fac)
    for i in range(2, len(params), 4):
        amplitude = params[i]
        pos = params[i + 1]
        sig = params[i + 2]
        gam = params[i + 3]
        y += voigt(x, amplitude, pos, sig, gam)
    return y


def multi_gaussian(x, *params):
    a = params[0]
    b = params[1]
    # beta = params[1]
    # c = params[2]
    # y = exp_dec(x, a, beta, c)
    y = a*x+b
    fac = (1 - np.sign(np.sign(x) + 1))
    det_bac = np.exp(x / (0.08617 * T) * fac)
    for i in range(2, len(params), 3):
        amplitude = params[i]
        mean = params[i + 1]
        fwhm = params[i + 2]
        # y += gaussian(x, amplitude, mean, fwhm)
        y += gaussian(x, amplitude, mean, fwhm)
    return y

def multi_lor(x, *params):
    # a = params[0]
    # beta = params[1]
    # c = params[2]
    # y = exp_dec(x, a, beta, c)
    k = params[0]
    b = params[1]
    y = k*x+b
    fac = (1 - np.sign(np.sign(x) + 1))
    det_bac = np.exp(x / (0.08617 * T) * fac)
    for i in range(2, len(params), 3):
        amplitude = params[i]
        mean = params[i + 1]
        gamma = params[i + 2]
        # y += gaussian(x, amplitude, mean, fwhm)
        y += lorentzian(x, amplitude, mean, gamma)*det_bac
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

print('============ Ei = 3.31 meV============')
path = '../data/ASCII_file/'
ratio_list = []
T = 40
Ei = 3.32
file = 'Ei{}_T{}.iexy'.format('3p32', T)
q, e, data, mask_i, mask_di = load_data(path, file)
energeRange = [0, 10]
# energeRange = [0, 0.03]
q_range = [1.65,1.75]
# q_range = [0.9,1.2]
# e_range = [-0.48, 0.54]
e_range = [-0.4, 0.4]
# e_range = [-1, 1]

config, conax = contourPlot(q, e, mask_i, energeRange)
conax.set_ylim(bottom=-0.1)
conax.set_xlim(right=2.3)
conax.vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
conax.vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
conax.hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
conax.hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')

x_q, sumII_e, err_e = disp(q, e, mask_i, mask_di, q_range, e_range)
fig1, ax1 = plt.subplots()
ax1.errorbar(x_q, sumII_e, err_e, marker='.', ls='none', fmt='o-', mfc='None', label='data')

ax1.set_ylabel('$\\rm I$ (a.u.)')
# #
# # fit_range = [1,1.7]
# fit_range =e_range
# p0=[1,1,
#     1, 0, 0.1
#     ]
# fit_x, fit_y, err_fit, pop, unc = fit_peak(x_q, sumII_e, err_e, p0, fit_range, fitfun=multi_gaussian)
# print('fitting parameters:\n{}'.format(pop))
# area1 =pop[2]
# print('fitting area:\n{}'.format(area1))
# print('uncertainties:\n{}'.format(unc))
# ax1.plot(fit_x, multi_gaussian(fit_x, *pop))

fit_range = e_range
ind = np.where(np.logical_and(x_q >= fit_range[0], x_q <= fit_range[1]))[0]
x_fit = np.array([i for i in x_q[ind]])
y_fit = np.array([i for i in sumII_e[ind]])
err_fit = np.array([i for i in err_e[ind]])

peak = SkewedGaussianModel()
background = LinearModel()
model = peak + background
pars = background.make_params(intercept = 3e-3, slope = 1e-3)
pars += peak.make_params(amplitude= 0.2,center = 0, sigma=0.5,gamma=0.5)

# pars = model.guess(y_fit, x=x_fit)
out = model.fit(y_fit, pars, x=x_fit)

print(out.fit_report(min_correl=0.25))
# fig1,ax1 = plt.subplots()
# ax1.plot(x_fit, y_fit)
ax1.plot(x_fit, out.best_fit, 'o-', label='best fit')
# area = out.
# plt.show()
print('============ Ei = 12 meV============')
####----------------------------------
### --------another Ei---------------
####---------------------------------

Ei = 12
file2 = 'Ei{}_T{}.iexy'.format(Ei, T)
q2, e2, data2, mask_i2, mask_di2 = load_data(path, file2)
# e_range = [-0.48, 0.54]
# e_range = [-0.8, 0.8]
##---------------
energeRange2 = [0, 0.5]
# e_range=[-0.6,0.6]
e_range = [-2, 2]
config, conax = contourPlot(q2, e2, mask_i2, energeRange2)
#### mask_i2 axis_1 = q, axis_2 = e
conax.hlines(3, xmin=(min(q2)), xmax=(max(q2)), ls='--', color='yellow')
conax.set_ylim(bottom=-0.5)
conax.set_xlim(right=4.5)
conax.vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
conax.vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
conax.hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
conax.hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')


x_q2, sumII_e2, err_e2 = disp(q2, e2, mask_i2, mask_di2, q_range, e_range)
# x_q2, sumII_e2, err_e2 = disp(q2, e2, mask_i2, mask_di2, q_range, e_range)
fig2, ax2 = plt.subplots()
ax2.errorbar(x_q2, sumII_e2, err_e2, marker='.', ls='none', fmt='o-', mfc='None', label='data')
ax2.set_ylabel('$\\rm I$ (a.u.)',fontsize=14)

# fit_range2 = e_range
# # fit_range2 = [-0.25,1]
# p02 = [1, 1,
#       0.5, 0, 0.5
#       ]
# fit_x2, fit_y2, err_fit2, pop2, unc2 = fit_peak(x_q2, sumII_e2, err_e2, p02, fit_range2, bounds=(0, np.inf), fitfun=multi_gaussian)
# print('fitting parameters:\n{}'.format(pop2))
# print('uncertainties:\n{}'.format(unc2))
# area2 =pop2[2]
# print('fitting area:\n{}'.format(area2))
# ax2.plot(fit_x2, multi_gaussian(fit_x2, *pop2), zorder=10, label='fit to data')
fit_range_2 = e_range
ind = np.where(np.logical_and(x_q2 >= fit_range_2[0], x_q2 <= fit_range_2[1]))[0]
x_fit = np.array([i for i in x_q2[ind]])
y_fit = np.array([i for i in sumII_e2[ind]])
err_fit = np.array([i for i in err_e2[ind]])

peak = SkewedGaussianModel()
background = LinearModel()
model = peak + background
pars = background.make_params(intercept = 3e-3, slope = 1e-3)
pars += peak.make_params(amplitude=0.2,center = 0, sigma=0.1,gamma=0.1)

# pars = model.guess(y_fit, x=x_fit)
out = model.fit(y_fit, pars, x=x_fit)

print(out.fit_report(min_correl=0.25))

plt.plot(x_fit, out.best_fit, 'o-', label='best fit')
plt.legend()
plt.show()
# print('ratio: {}'.format(rat))


# # ##--------------------------------------------------
# text = ''
# fit_range = [3, 10]
# num_of_peaks = 2
#
# ind = np.where(np.logical_and(x_e >= fit_range[0], x_e <= fit_range[1]))[0]
# x_fit = np.array([i for i in x_e[ind]])
# y_fit = np.array([i for i in sumII_q[ind]])
# err_fit = np.array([i for i in err_q[ind]])
# ax1.vlines(min(fit_range), ymin=0, ymax=max(sumII_q), ls='--', color='r')
# ax1.vlines(max(fit_range), ymin=0, ymax=max(sumII_q), ls='--', color='r')
# # fmod = Model(exp_dec, prefix='bg_')
# fmod = Model(linear, prefix='bg_')
# for i in range(num_of_peaks):
#     fmod += Model(voigt, prefix='l{}_'.format(i + 1))
# params = fmod.make_params()
# print('************************************')
# print('Start fitting...')
# print(f'Parameter names: {fmod.param_names}')
#
# params['bg_slope'].set(0)
# params['bg_intercept'].set(0,min=0)
# # params['bg_slope'].set(vary=False)
#
# params['l1_amplitude'].set(0.001,min=0)
# params['l1_mean'].set(5)
# params['l1_fwhm'].set(ins_res_12(4.894),vary=False)
# params['l1_gamma'].set(0.4)
#
# params['l2_amplitude'].set(0.001,min=0)
# params['l2_mean'].set(8)
# params['l2_fwhm'].set(ins_res_12(7.961),vary=False)
# params['l2_gamma'].set(0.4)
#
# result = fmod.fit(y_fit, params, x=x_fit)
# # print(result.fit_report())
# ax1.plot(x_fit, result.best_fit)
# ax1.set_ylim(bottom = 0,top=0.02)
# comps = result.eval_components(x=x_fit)
# print('************************************')
# print('Fitting is done!')
#
# for name, par in result.params.items():
#      print("  %s: value=%f +/- %f " % (name, par.value, par.stderr))
#
#
# fig2, ax2 = plt.subplots()
# ax2.plot(x_fit,y_fit-comps['bg_'],marker='.', ls='none', mfc='None', label='data')
# ax2.plot(x_fit,result.best_fit-comps['bg_'],label='fit to data')
# for i in range(num_of_peaks):
#     ax2.plot(x_fit, comps['l{}_'.format(i+1)], label=r'Gaussian #{}'.format(i+1))
