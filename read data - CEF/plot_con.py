import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
from scipy.optimize import curve_fit
import scipy.special as spf
import load_data as ld
import timeit
import matplotlib
matplotlib.use("QtAgg")
import scipy.constants as cts


def contourPlot(q, e, int, int_ran):
    fig, ax = plt.subplots(dpi=300)
    # fig, ax = plt.subplots()
    cp = ax.pcolormesh(q, e, int, cmap='jet', vmin=int_ran[0], vmax=int_ran[1])
    cbar = fig.colorbar(mappable=cp)
    # ax.set_title("cplot_{}K_Ei{}".format(T, Ei),size=20)
    ax.set_xlabel("Q $(Å^{-1})$", size=25, labelpad=2)
    ax.set_ylabel("$\\hbar \\omega$ (meV)", size=25)
    ax.vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    ax.vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    ax.hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    ax.hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')

    ax.set_xticks(np.arange(0.5, 4.6, 1))
    ax.set_yticks(np.arange(-2, 13, 2))
    ax.tick_params(axis='both', labelsize=25)
    cbar.ax.tick_params(labelsize=25)
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


def gaussian(x, amplitude, mean, stddev):
    y = amplitude * np.exp(-(x - mean) ** 2 / 2 / (stddev ** 2))
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
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        y += gaussian(x, amplitude, mean, sigma)
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
    return fig, ax


# ####--------------------------------------------
folder = '../data/ASCII_file/'
ei_list = [12,3.32]
t_list= [10,30,40,60]
data_list = []

start = timeit.default_timer()

for ei in ei_list:
    for temp in t_list:
        data_list.append(ld.data(folder, ei, temp))
stop = timeit.default_timer()
print('Inport data sets in {} seconds '.format(stop-start)) ### 0-4 3.32meV, 5-9 12meV


intensity = [0, 0.01]
q_range = [1.2, 2.4]
e_range = [3, 10]

config, axs = plt.subplots(2,5,figsize=(6.75,3.4),gridspec_kw={'width_ratios': [5,5,5,5,1.5]})
for idx, data in enumerate(data_list):
    q = data.q
    e = data.e
    int = data.mI
    if idx<=3:
        axs[0][idx].pcolormesh(q, e, int, cmap='jet', vmin=intensity[0], vmax=intensity[1])
        axs[0][idx].set_title("T = {}K".format(data.temp),fontsize=10)
        axs[0][idx].tick_params(axis='both', labelsize=10)
        axs[0][idx].set_xticks(np.arange(0.5, 4.6, 1.5))
        axs[0][idx].set_yticks(np.arange(-3, 13,3))
        axs[0][idx].set_ylim(bottom=-0.5,top=11.5)
        axs[0][idx].set_xlim(left=0.4,right=4.5)
    else:
        cp=axs[1][idx-4].pcolormesh(q, e, int, cmap='jet', vmin=intensity[0], vmax=intensity[1])
        axs[1][idx-4].set_xticks(np.arange(0.5, 4.6, 1))
        axs[1][idx-4].set_yticks(np.arange(-2, 3.2, 1))
        axs[1][idx-4].tick_params(axis='both', labelsize=10)
        axs[1][idx-4].set_ylim(bottom=-0.1,top=3)
        axs[1][idx-4].set_xlim(left=0.2,right=2.5)
axs[0][4].xaxis.set_visible(False)
axs[0][4].yaxis.set_visible(False)
axs[1][4].xaxis.set_visible(False)
axs[1][4].yaxis.set_visible(False)
axs[0][4].spines['top'].set_visible(False)
axs[0][4].spines['right'].set_visible(False)
axs[0][4].spines['bottom'].set_visible(False)
axs[0][4].spines['left'].set_visible(False)
axs[1][4].spines['top'].set_visible(False)
axs[1][4].spines['right'].set_visible(False)
axs[1][4].spines['bottom'].set_visible(False)
axs[1][4].spines['left'].set_visible(False)
# cbar = config.colorbar(mappable=cp)
# config.subplots_adjust(right=15)
cbar_ax1 = config.add_axes([0.88, 0.61, 0.02, 0.29])
cbar_ax2 = config.add_axes([0.88, 0.2, 0.02, 0.29])

config.colorbar(cp,cax=cbar_ax1)
cbar_ax1.tick_params(labelsize=10)
config.colorbar(cp,cax=cbar_ax2)
cbar_ax2.tick_params(labelsize=10)

# axs.set_xticks(np.arange(0.5, 4.6, 1))
# axs.set_yticks(np.arange(-2, 13, 2))
# axs.tick_params(axis='both', labelsize=25)
# cbar.ax.tick_params(labelsize=25)
# plt.xlabel("Q $(Å^{-1})$", size=25)
# config.set_ylabel("$\\hbar \\omega$ (meV)", size=25)
config.supxlabel("Q $(Å^{-1})$", size=10,y=0.09)
config.supylabel("$\\hbar \\omega$ (meV)", size=10,x=0.04)
config.tight_layout()
# config.subplots_adjust(wspace=0.5, hspace=0.5)
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.show()

# plt.savefig("testcon.png",dpi=600)

# cbar = fig.colorbar(mappable=cp)
# ax.set_xlabel("Q $(Å^{-1})$", size=25)
# ax.set_ylabel("$\\hbar \\omega$ (meV)", size=25)
# ax.vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
# ax.vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
# ax.hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
# ax.hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
# ax.set_xticks(np.arange(0.5, 4.6, 1))
# ax.set_yticks(np.arange(-2, 13, 2))
# ax.tick_params(axis='both', labelsize=25)
# cbar.ax.tick_params(labelsize=25)
# ax.set_ylim(bottom=-0.5)
# ax.set_xlim(right=4.5)
# fig.tight_layout()


# plt.show()
