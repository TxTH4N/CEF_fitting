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


def contourPlot(q, e, int, int_ran,q_range,e_range):
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
    sumII = np.nanmean(IIMat, axis=0)
    err_sq = np.nansum(np.square(errMat), axis=0)
    n = np.sum(1 - np.isnan(errMat), axis=0)
    err = np.sqrt(err_sq) / n
    return qx, sumII, err

def linear(x,slope,intercept):
    y = slope*x+intercept
    return y

def gaussian(x, amplitude, mean, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    y = amplitude/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x - mean) ** 2 / 2 / (sigma ** 2))
    return y


# def voigt(x, amplitude, mean, fwhm, gamma):
#     sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
#     voi = amplitude * spf.voigt_profile(x - mean, sigma, gamma)     ### gamma is half-width
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

def multi_voi(x, *params):
    a = params[0]
    beta = params[1]
    c = params[2]
    y = exp_dec(x, a, beta, c)
    for i in range(3, len(params), 4):
        area = params[i]
        pos = params[i + 1]
        sig = params[i + 2]
        gam = params[i + 3]
        y += voigt(x, area, pos, sig, gam)     ### gamma is half-width
    return y


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
path = '../data/ASCII_file/'
Ei = 12
T = 60
file = 'Ei{}_T{}.iexy'.format('12', T)
q, e, data, mask_i, mask_di = load_data(path, file)

##---------------

energeRange = [0, 0.01]
# energeRange = [0, 10]

path = '../data/ASCII_file/'
Ei2 = 12
T2 = 300
file2 = 'Ei{}_T{}.iexy'.format('12', T2)
q2, e2, data2, mask_i2, mask_di2 = load_data(path, file2)


### ----------------------
q_range3 = [1,4]  ## q bin range
# q_range = [1.6,1.8]
e_range3 = [-0.4,0.4]  ## plot range
config3, conax3 = contourPlot(q2, e2, mask_i2-mask_i, energeRange,q_range3,e_range3)
conax3.set_ylim(bottom=-0.5)
conax3.set_xlim(right=4.5)
#
x_q3, sumII_e3, err_e3 = disp(q2, e2, mask_i2-mask_i, np.sqrt(mask_di2**2+mask_di**2), q_range3, e_range3)
#
fig3, ax3 = plt.subplots()

ax3.errorbar(x_q3, sumII_e3, err_e3, marker='.', ls='-', fmt='o-', mfc='None', label='data')
# ax3.set_xlabel('$\\hbar \\omega$ (meV)')
# ax3.set_ylabel('$\\rm I$ (a.u.)')
# ax3.set_title('Energy dispersion')
# ax3.set_ylim(bottom=0,top=0.015)
# ax3.set_xlim(left=0,right=3)

plt.show()

