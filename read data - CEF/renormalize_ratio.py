import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
from scipy.optimize import curve_fit
import scipy.special as spf
import matplotlib
matplotlib.use('QtAgg')

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


def gaussian(x, amplitude, mean, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    y = amplitude/np.sqrt(2*np.pi)/sigma * np.exp(-(x - mean) ** 2 / 2 / (sigma ** 2))
    return y

def lorentzian(x, amplitude, mean, gamma):
    y = amplitude / np.pi*0.5*gamma/((x-mean)**2+(0.5*gamma)**2)
    return y

def voigt(x, amp, pos, sig, gam):
    voi = amp*spf.voigt_profile(x - pos, sig, gam)
    return voi


def exp_dec(x, a, beta, c):
    y = a * np.exp(-x * beta) + c
    return y


def multi_voi(x, *params):
    a = params[0]
    beta = params[1]
    c = params[2]
    y = exp_dec(x, a, beta, c)
    fac = (1 - np.sign(np.sign(x) + 1))
    det_bac = np.exp(x / (0.08617 * T) * fac)
    for i in range(3, len(params), 4):
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
        y += gaussian(x, amplitude, mean, fwhm)*det_bac
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
    # det_bac = np.exp(x / (0.08617 * T) * fac)
    for i in range(2, len(params), 3):
        amplitude = params[i]
        mean = params[i + 1]
        gamma = params[i + 2]
        # y += gaussian(x, amplitude, mean, fwhm)
        y += lorentzian(x, amplitude, mean, gamma)
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

def cal_ratio(T):
    path = '../data/ASCII_file/'
    ratio_list = []
    Ei = 3.32
    file = 'Ei{}_T{}.iexy'.format('3p32', T)
    q, e, data, mask_i, mask_di = load_data(path, file)
    energeRange = [0, 10]
    # energeRange = [0, 0.03]
    # q_range = [1,1.7]
    q_range = [1.7,2.0]
    # e_range = [-0.48, 0.54]
    e_range = [-0.4, 0.4]

    config, conax = contourPlot(q, e, mask_i, energeRange)
    conax.set_ylim(bottom=-0.1)
    conax.set_xlim(right=2.3)
    conax.vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    conax.vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    conax.hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    conax.hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')

    x_q, sumII_e, err_e = dispQ(q, e, mask_i, mask_di, q_range, e_range)
    fig1, ax1 = plt.subplots()
    ax1.errorbar(x_q, sumII_e, err_e, marker='.', ls='none', fmt='o-', mfc='None', label='data')
    # ax1.set_xlabel('Q ($\AA^{-1}$)')
    # ax1.set_xlabel('E (meV)')
    ax1.set_ylabel('$\\rm I$ (a.u.)')
    # ax1.set_title('Energy dispersion')

    # fit_range = [1,1.7]
    fit_range = [1.7, 2.0]
    p0=[1,1,
        0.5, 1.9, 0.05
        ]
    # p0=[1,1,
    #     0.5, 1.9, 0.05,
    #     ]
    fit_x, fit_y, err_fit, pop, unc = fit_peak(x_q, sumII_e, err_e, p0, fit_range, fitfun=multi_lor)
    print('fitting parameters:\n{}'.format(pop))
    area1 =pop[2]
    print('fitting area:\n{}'.format(area1))
    print('uncertainties:\n{}'.format(unc))
    xx = np.linspace(1.7,2,300)
    ax1.plot(xx, multi_lor(xx, *pop))



    ####----------------------------------
    ### --------another Ei---------------
    ####---------------------------------

    Ei = 12
    file2 = 'Ei{}_T{}.iexy'.format(Ei, T)
    q2, e2, data2, mask_i2, mask_di2 = load_data(path, file2)
    # e_range = [-0.48, 0.54]
    ##---------------
    energeRange2 = [0, 1]
    config, conax = contourPlot(q2, e2, mask_i2, energeRange2)
    #### mask_i2 axis_1 = q, axis_2 = e
    conax.hlines(3, xmin=(min(q2)), xmax=(max(q2)), ls='--', color='yellow')
    conax.set_ylim(bottom=-0.5)
    conax.set_xlim(right=4.5)
    conax.vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    conax.vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    conax.hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    conax.hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')


    x_q2, sumII_e2, err_e2 = dispQ(q2, e2, mask_i2, mask_di2, q_range, e_range)
    # x_q2, sumII_e2, err_e2 = disp(q2, e2, mask_i2, mask_di2, q_range, e_range)
    fig2, ax2 = plt.subplots()
    ax2.errorbar(x_q2, sumII_e2, err_e2, marker='.', ls='none', fmt='o-', mfc='None', label='data')
    # ax2.set_xlabel('Q ($\AA^{-1}$)')
    # ax2.set_xlabel('E (meV)')
    ax2.set_ylabel('$\\rm I$ (a.u.)',fontsize=14)
    # ax2.set_title('Energy dispersion')

    # fit_range2 = [1,1.7]
    fit_range2 = [1.7,2.0]
    p02=[10,5,
        1, 1.9, 0.05,
        ]


    fit_x2, fit_y2, err_fit2, pop2, unc2 = fit_peak(x_q2, sumII_e2, err_e2, p02, fit_range2, bounds=(0, np.inf), fitfun=multi_lor)
    print('fitting parameters:\n{}'.format(pop2))
    print('uncertainties:\n{}'.format(unc2))
    area2 =pop2[2]
    print('fitting area:\n{}'.format(area2))
    ax2.plot(xx, multi_lor(xx, *pop2), zorder=10, label='fit to data')


    plt.legend()
    # plt.show()

    # normalize with (1 0 2)
    tth_3p32 = 96.76*np.pi/180
    tth_12 = 42.6*np.pi/180

    # normalize with (1 0 2)
    tth_3p32 = 96.76*np.pi/180
    tth_12 = 42.6*np.pi/180

    wl_3p32 = 9.044567/np.sqrt(3.32)
    wl_12 = 9.044567/np.sqrt(12)
    # rat = area1 / area2 *(wl_12/wl_3p32)**4*(np.sin(tth_3p32)/np.sin(tth_12))*(np.sin(tth_3p32/2)/np.sin(tth_12/2))
    rat = area1 / area2 * (wl_12 / wl_3p32) ** 4 * (np.sin(tth_3p32/2) / np.sin(tth_12/2))**3
    # rat = area1 / area2 * (wl_12 / wl_3p32) ** 4 * (
    #             np.sin(tth_3p32 / 2)**2 / np.sin(tth_12 / 2)**2)
    # print('ratio: {}'.format(rat))
    return rat

# ####--------------------------------------------

if __name__ == '__main__':
    T_list = [10,30,40,60]
    rat_list=[]
    for idx,T in enumerate(T_list):
        rat_list.append(cal_ratio(T))
    print("The list of ratio of peak (1 0 2)")
    print(rat_list)
    plt.show()
    print(np.average(rat_list))
