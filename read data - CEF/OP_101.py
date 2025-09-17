import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
from scipy.optimize import curve_fit
from lmfit import Model, Parameters


def linear_bg(x, slope, intercept):
    y = slope * x + intercept
    return y


def lorentzian(x, amplitude, mean, gamma):
    y = amplitude / np.pi * 0.5 * gamma / ((x - mean) ** 2 + (0.5 * gamma) ** 2)
    return y


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
    # fig, ax = plt.subplots(figsize = (4,3))
    fig, ax = plt.subplots(figsize=(12, 4))
    cp = ax.pcolormesh(q, e, int, cmap='jet', vmin=int_ran[0], vmax=int_ran[1])
    fig.colorbar(mappable=cp)
    ax.vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    ax.vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    ax.hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    ax.hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    # fig.show()
    return fig, ax


def disp(q, e, ii, di, Q_range, E_range, sumE=False):
    q_ind = np.where(np.logical_and(q >= Q_range[0], q <= Q_range[1]))[0]  ## get q bin index
    e_ind = np.where(np.logical_and(e >= E_range[0], e <= E_range[1]))[0]
    ex = np.array([ei for ei in e[e_ind]])
    qx = np.array([qi for qi in q[q_ind]])
    IIMat = ii[e_ind, :][:, q_ind]
    errMat = di[e_ind, :][:, q_ind]
    if sumE is False:
        sumII = np.nanmean(IIMat, axis=1)
        err_sq = np.nansum(np.square(errMat), axis=1)
        n = np.sum(1 - np.isnan(errMat), axis=1)
        err = np.sqrt(err_sq) / n
        return ex, sumII, err
    elif sumE is True:
        sumII = np.nanmean(IIMat, axis=0)
        err_sq = np.nansum(np.square(errMat), axis=0)
        n = np.sum(1 - np.isnan(errMat), axis=0)
        err = np.sqrt(err_sq) / n
        return qx, sumII, err


def int(q, e, ii, di, Q_range, E_range):
    q_ind = np.where(np.logical_and(q >= Q_range[0], q <= Q_range[1]))[0]  ## get q bin index
    e_ind = np.where(np.logical_and(e >= E_range[0], e <= E_range[1]))[0]
    ex = np.array([ei for ei in e[e_ind]])
    qx = np.array([qi for qi in q[q_ind]])
    IIMat = ii[e_ind, :][:, q_ind]
    errMat = di[e_ind, :][:, q_ind]
    sumII = np.nansum(IIMat, axis=1)
    sum = np.nansum(sumII)
    err_sq = np.nansum(np.square(errMat), axis=1)
    err = np.sqrt(np.nansum(err_sq))
    return sum, err


path = '../data/ASCII_file/Q0p01/'
q_range = [1.42, 1.55]  ## q bin range
# q_range = [1.8,2]
# q_range = [1.3, 1.42]

e_range = [-0.15, 0.15]  ## plot range
Ei = '3p32'
# bg_10K = 'Ei{}_T{}.iexy'.format(Ei, 10)
# q_bg, e_bg, data_bg, mask_i_bg, mask_di_bg = load_data(path, bg_10K)
# sumII, err = int(q_bg, e_bg, mask_i_bg, mask_di_bg, q_range, e_range)
# print(sumII)

file_list = [0.25, 1.5, 1.7, 3, 3.5, 3.7, 4, 5, 6]
ii_list = []
err_list = []
fig, ax = plt.subplots()
II = []
II_err = []

for idx, T in enumerate(file_list):
    temp_str = str(T)
    if '.' in temp_str:
        temp_str = temp_str.replace('.', 'p')
    file = 'Ei{}_T{}.iexy'.format('3p32', temp_str)
    q, e, data, mask_i, mask_di = load_data(path, file)

    config, conax = contourPlot(q, e, mask_i, [0, 10])
    conax.set_ylim(bottom=-0.2, top=0.2)
    conax.set_xlim(left=1.2, right=2.4)

    # sumII, err = int(q, e, mask_i, mask_di, q_range, e_range)
    # ii_list.append(sumII)
    # err_list.append(err)
    qx, sumII_q, err_q = disp(q, e, mask_i, mask_di, q_range, e_range, sumE=True)
    ax.errorbar(qx, sumII_q, yerr=err_q, fmt='o')

    if idx >= 7:
        II.append(0)
        II_err.append(0)
        continue
    # Fit the spectrum below
    fit_range = [1.42, 1.55]
    plot_x_range = np.linspace(fit_range[0], fit_range[1], 100)
    num_of_peaks = 1
    ind = np.where(np.logical_and(qx >= fit_range[0], qx <= fit_range[1]))[0]
    x_fit = np.array([i for i in qx[ind]])
    y_fit = np.array([i for i in sumII_q[ind]])
    err_fit = np.array([i for i in err_q[ind]])
    fmod = Model(linear_bg, prefix='bg_')
    for i in range(num_of_peaks):
        fmod += Model(lorentzian, prefix='p{}_'.format(i + 1))
    params = fmod.make_params()
    print('************************************')
    print('----------Start fitting----------')
    params['bg_slope'].set(0.1)
    params['bg_intercept'].set(0.1)
    # print(params.__dir__())
    for i in range(num_of_peaks):
        params['p{}_amplitude'.format(i + 1)].set(0.2, min=0)
        params['p{}_mean'.format(i + 1)].set(1.5)
        # params['p{}_fwhm'.format(i + 1)].set(0.3)
        params['p{}_gamma'.format(i + 1)].set(0.1)
    result = fmod.fit(y_fit, params, x=x_fit, weights=1 / err_fit)
    # result = fmod.fit(y_fit, params, x=x_fit)
    print(result.fit_report(min_correl=0.25))
    comps = result.eval_components(x=x_fit)
    print('----------Fitting is done!----------')
    print('----------↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓----------')
    fit_params = []
    for name, par in result.params.items():
        print("  %s: value=%f +/- %f " % (name, par.value, par.stderr))
        fit_params.append(par.value)
    ax.plot(x_fit, result.best_fit, linestyle='--', color='pink')
    ax.plot(plot_x_range, linear_bg(plot_x_range, *fit_params[:2]) + lorentzian(plot_x_range, *fit_params[2:5]))
    II.append(result.result.params['p1_amplitude'].value)
    II_err.append(result.result.params['p1_amplitude'].stderr)
print()
print(II)
print(II_err)

fig, ax = plt.subplots()
ax.errorbar(file_list, II, II_err, marker='.', ls='none', fmt='o-', mfc='None', label='data')
ax.set_xlabel('T (K)', fontsize=15)
ax.set_ylabel('II (a.u.)', fontsize=15)
ax.set_title('Integrated intensity on (1 0 1)', fontsize=15)
ax.tick_params(axis='both', labelsize=13)

plt.show()
