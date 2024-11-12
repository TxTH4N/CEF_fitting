import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')
import PyCrystalField as cef
from scipy.optimize import curve_fit
import scipy.special as spf

import scipy.constants as cts
from lmfit import Model, Parameters


# from lmfit.models import GaussianModel,ExponentialModel


# ####--------crystal field calculation---------------
def cry_fie_cal(Bdictionary):
    ion = 'Tb3+'
    TVS = cef.CFLevels.Bdict(ion, Bdictionary)
    TVS.diagonalize()
    print('********************************************')
    print('Eigenvalues of Hamiltonian based on provided B:\n{}'.format(TVS.eigenvalues))
    TVS.printEigenvectors()
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
    return fig,ax


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

def linear(x,slope,intercept):
    y = slope*x+intercept
    return y

def gaussian(x, amplitude, mean, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    y = amplitude/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x - mean) ** 2 / 2 / (sigma ** 2))
    return y


# def voigt(x, amplitude, mean, fwhm, gamma):
#     sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
#     voi = amplitude * spf.voigt_profile(x - mean, sigma, gamma/2)     ### gamma is full-width
#     return voi

def voigt( x,amplitude, mean, fwhm, gamma):
    """ Return the Voigt line shape at x with Lorentzian component FWHM gamma
    and Gaussian component FWHM alpha."""
    sigma = (0.5*fwhm) / np.sqrt(2 * np.log(2))
    return amplitude*np.real(spf.wofz(((x-mean) + 1j*(0.5*gamma))/sigma/np.sqrt(2))) / sigma\
                                                            /np.sqrt(2*np.pi)

def exp_dec(x, a, beta, c):
    y = a * np.exp(-x * beta) + c
    return y

def lorentzian(x, amplitude, mean, gamma):
    y = amplitude / np.pi*0.5*gamma/((x-mean)**2+(0.5*gamma)**2)    ### gamma is full-width
    return y

# def multi_voi(x, *params):
#     a = params[0]
#     beta = params[1]
#     c = params[2]
#     y = exp_dec(x, a, beta, c)
#     for i in range(3, len(params), 4):
#         area = params[i]
#         pos = params[i + 1]
#         sig = params[i + 2]
#         gam = params[i + 3]/2
#         y += voigt(x, area, pos, sig, gam)     ### gamma is half-width
#     return y


def multi_gaussian(x, *params):
    a = params[0]
    beta = params[1]
    c = params[2]
    y = exp_dec(x, a, beta, c)
    for i in range(3, len(params), 3):
        area = params[i]
        mean = params[i + 1]
        fwhm = params[i + 2]
        y += gaussian(x, area, mean, fwhm)
    return y

def ins_res_3p32(x):
    y = 0.00030354 * x ** 3 + 0.0039009 * x ** 2 - 0.040862 * x + 0.11303
    return y

def ins_res_12(x):
    y =  3.8775e-05 * x**3 +0.0018964 * x**2 -0.078275 * x +0.72305
    return y


def fit_peak(x, y, err, p0, x_range=[-np.inf, np.inf], bounds=(0, np.inf), fitfun=multi_gaussian):
    ind = np.where(np.logical_and(x >= x_range[0], x <= x_range[1]))[0]
    x_fit = np.array([i for i in x[ind]])
    y_fit = np.array([i for i in y[ind]])
    err_fit = np.array([i for i in err[ind]])
    pop, pco = curve_fit(fitfun, x_fit, y_fit, p0=p0, bounds=bounds, sigma=err_fit, absolute_sigma=True)
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


# ####--------------------------------------------
path = '../../data/ASCII_file/'
Ei = 12
T = 60
file = 'Ei{}_T{}.iexy'.format('12', T)
q, e, data, mask_i, mask_di = load_data(path, file)

##---------------

energeRange = [0, 0.01]
# energeRange = [-4, -2]
# energeRange = [0, 10]

# # ##------ the intensity is average of specific energy level
q_range = [1.3,1.9]  ## q bin range
# q_range = [1.6,1.8]
e_range = [3, 10]  ## plot range



config, conax = contourPlot(q, e, mask_i, energeRange)
conax.set_ylim(bottom=-0.5)
conax.set_xlim(right=4.5)

x_e, sumII_q, err_q = disp(q, e, mask_i, mask_di, q_range, e_range)

fig, ax1 = plt.subplots()
ax1.errorbar(x_e, sumII_q, err_q, marker='.', ls='none', fmt='o-', mfc='None', label='data')
ax1.set_xlabel('$\\hbar \\omega$ (meV)')
ax1.set_ylabel('$\\rm I$ (a.u.)')
ax1.set_title('Energy dispersion')

# ##--------------------------------------------------
text = ''
fit_range = [3, 10]
num_of_peaks = 2

ind = np.where(np.logical_and(x_e >= fit_range[0], x_e <= fit_range[1]))[0]
x_fit = np.array([i for i in x_e[ind]])
y_fit = np.array([i for i in sumII_q[ind]])
err_fit = np.array([i for i in err_q[ind]])
ax1.vlines(min(fit_range), ymin=0, ymax=max(sumII_q), ls='--', color='r')
ax1.vlines(max(fit_range), ymin=0, ymax=max(sumII_q), ls='--', color='r')
# fmod = Model(exp_dec, prefix='bg_')
fmod = Model(linear, prefix='bg_')
for i in range(num_of_peaks):
    fmod += Model(voigt, prefix='v{}_'.format(i + 1))
params = fmod.make_params()
print('************************************')
print('Start fitting...')
print(f'Parameter names: {fmod.param_names}')

params['bg_slope'].set(0)
params['bg_intercept'].set(0,min=0)
# params['bg_slope'].set(vary=False)

params['v1_amplitude'].set(0.001,min=0)
params['v1_mean'].set(5)
params['v1_fwhm'].set(ins_res_12(4.894),vary=False)
params['v1_gamma'].set(0.4)

params['v2_amplitude'].set(0.001,min=0)
params['v2_mean'].set(8)
params['v2_fwhm'].set(ins_res_12(7.961),vary=False)
params['v2_gamma'].set(0.4)

result = fmod.fit(y_fit, params, x=x_fit)
# print(result.fit_report())
ax1.plot(x_fit, result.best_fit)
ax1.set_ylim(bottom = 0,top=0.02)
comps = result.eval_components(x=x_fit)
print('************************************')
print('Fitting is done!')

for name, par in result.params.items():
     print("  %s: value=%f +/- %f " % (name, par.value, par.stderr))


fig2, ax2 = plt.subplots()
ax2.plot(x_fit,y_fit-comps['bg_'],marker='.', ls='none', mfc='None', label='data')
ax2.plot(x_fit,result.best_fit-comps['bg_'],label='fit to data')
for i in range(num_of_peaks):
    ax2.plot(x_fit, comps['v{}_'.format(i+1)], label=r'Gaussian #{}'.format(i+1))

# # print("-----------B factor  calculation --------------")

# # # adding B60 - case I
B20 = -1.02994297e-01
B40 = -5.94668808e-04
B60 = 2.31716153e-06
B66=-1.64771401e-06

fig3, ax3 = plt.subplots()

Bdictionary = {'B20': B20, 'B40': B40,
               'B60': B60, 'B66': B66}
# #
TVS = cry_fie_cal(Bdictionary)
x_1 = np.linspace(0,3.32,1000)
x_2 = np.linspace(3.32,12,1000)
sim_int_40_3p32 = 0.00273018*TVS.normalizedNeutronSpectrum(x_1, Temp=60, ResFunc=ins_res_3p32,gamma=0.2)
sim_int_40_12 = 0.00273018*TVS.normalizedNeutronSpectrum(x_2, Temp=60, ResFunc=ins_res_12,gamma=0.2)
ax3.plot(x_1, sim_int_40_3p32, label='parameter1_40K_B66{}'.format(B66), marker='.',markersize=2)
ax3.plot(x_2, sim_int_40_12, label='parameter1_40K_B66{}'.format(B66), marker='.',markersize=2)
ax3.plot(x_fit,y_fit)
ax3.set_xlabel('$\\hbar \\omega$ (meV)')
ax3.set_ylabel('$\\rm I$ (a.u.)')
ax3.set_ylim(bottom=0,top=0.05)

plt.show()
