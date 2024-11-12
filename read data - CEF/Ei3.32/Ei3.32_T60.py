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
def point_charge():
    Tb166LigP, TbP = cef.importCIF('../TbV6Sn6_edit.cif', 'Tb1', Zaxis=[0, 0, 1], MaxDistance=5.5)
    # TbP.printEigenvectors()
    TbP.diagonalize()
    print("Eigenvalues of the point charge model:")
    print(TbP.eigenvalues)
    CalculatedSpectrum = TbP.normalizedNeutronSpectrum(x_e, Temp=2, ResFunc=lambda x: 0.1)
    fig, ax = plt.subplots()
    ax.plot(x_e, CalculatedSpectrum, label='point charge model')
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


def gaussian(x, area, mean, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    y = area/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x - mean) ** 2 / 2 / (sigma ** 2))
    return y


def voigt(x, amp, pos, sig, gam):
    voi = amp * spf.voigt_profile(x - pos, sig, gam)
    return voi


def exp_dec(x, a, beta, c):
    y = a * np.exp(-x * beta) + c
    return y

def lorentzian_bg(x, amplitude, mean, gamma,const):
    y = amplitude / np.pi*0.5*gamma/((x-mean)**2+(0.5*gamma)**2)+const
    return y

def lorentzian(x, amplitude, mean, gamma):
    y = amplitude / np.pi*0.5*gamma/((x-mean)**2+(0.5*gamma)**2)
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
        y += voigt(x, area, pos, sig, gam)
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


# def plot(x, y, err):
#     fig, ax = plt.subplots()
#     ax.errorbar(x, y, err, marker='.', ls='none', fmt='o-', mfc='None', label='data')
#     ax.set_xlabel('$\\hbar \\omega$ (meV)')
#     ax.set_ylabel('$\\rm I$ (a.u.)')
#     ax.set_title('Energy dispersion')
#     fig.show()
#     # ax.set_ylim(bottom=0)
#     # ax.set_xlim(left=0)
#     return fig, ax
# def fit_plot(x, y, para, err, fitfun=multi_gaussian):
#     f, ax = plot(x, y, err)
#     ax.plot(x, fitfun(x, *para), zorder=10, label='fit to data')
#     for i in range(3, len(para), 3):
#         hliney = 0.5 * para[i] + exp_dec(para[i], para[0], para[1], para[2])
#         ax.hlines(hliney, xmin=para[i] - ins_res(para[i], 12) / 2, xmax=para[i] + ins_res(para[i], 12) / 2,
#                   color='r',
#                   label='instrument resolution limit')
#     # hliney = 0.5 * pop[3] + exp_dec(pop[4], pop[0], pop[1])
#     # hliney2 = 0.5 * pop[6] + exp_dec(pop[7], pop[0], pop[1])
#     # ax.hlines(hliney, xmin=pop[4] - ins_res(pop[4], 3.32) / 2, xmax=pop[4] + ins_res(pop[4], 3.32) / 2, color='r',
#     #           label='instrument resolution limit')
#     # ax.hlines(hliney2, xmin=pop[7] - ins_res(pop[7], 3.32) / 2, xmax=pop[7] + ins_res(pop[7], 3.32) / 2, color='r')
#     plt.legend()
#     return 0


# ####--------------------------------------------
path = '../../data/ASCII_file/'
Ei = 3.32
T = 60
file = 'Ei{}_T{}.iexy'.format('3p32', T)
q, e, data, mask_i, mask_di = load_data(path, file)

##---------------

energeRange = [0, 0.01]
# energeRange = [0, 10]

# # ##------ the intensity is average of specific energy level

# q_range = [1, 1.6]  ## q bin range
q_range = [1.3,1.9]
e_range = [-0.1, 3]  ## plot range

config, conax = contourPlot(q, e, mask_i, energeRange)
conax.set_ylim(bottom=-0.1)
conax.set_xlim(right=2.3)

x_e, sumII_q, err_q = disp(q, e, mask_i, mask_di, q_range, e_range)
fig, ax1 = plt.subplots()
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
# fig, ax = plt.subplots()
# fit_range = [0.3, 1.07]
# fit_range = [1.5, 2]
# fit_range = [1.07, 1.5]
fit_range = [0.2, 2.7]
# fit_range = [0.3, 1]
num_of_peaks = 5

ind = np.where(np.logical_and(x_e >= fit_range[0], x_e <= fit_range[1]))[0]
x_fit = np.array([i for i in x_e[ind]])
y_fit = np.array([i for i in sumII_q[ind]])
err_fit = np.array([i for i in err_q[ind]])
ax1.vlines(min(fit_range), ymin=0, ymax=max(sumII_q), ls='--', color='r')
ax1.vlines(max(fit_range), ymin=0, ymax=max(sumII_q), ls='--', color='r')
# fmod = Model(exp_dec, prefix='bg_')
fmod = Model(lorentzian_bg, prefix='bg_')
for i in range(num_of_peaks):
    fmod += Model(lorentzian, prefix='l{}_'.format(i + 1))
params = fmod.make_params()
print('************************************')
print('Start fitting...')
print(f'Parameter names: {fmod.param_names}')

params['bg_amplitude'].set(4)
params['bg_mean'].set(0)
params['bg_gamma'].set(0.004)
params['bg_const'].set(0.0)

params['l1_amplitude'].set(0.001)
params['l1_mean'].set(0.7)
params['l1_gamma'].set(0.2)

params['l2_amplitude'].set(0.001)
params['l2_mean'].set(1.25)
params['l2_gamma'].set(0.2)

params['l3_amplitude'].set(0.001)
params['l3_mean'].set(1.7)
params['l3_gamma'].set(0.2)

params['l4_amplitude'].set(0.001)
params['l4_mean'].set(0.8)
params['l4_gamma'].set(expr="l1_gamma")

params['l5_amplitude'].set(0.001)
params['l5_mean'].set(1.6)
params['l5_gamma'].set(expr="l3_gamma")




result = fmod.fit(y_fit, params, x=x_fit)
# print(result.fit_report())
ax1.plot(x_fit, result.best_fit)
ax1.set_ylim(bottom = 0,top=0.01)
comps = result.eval_components(x=x_fit)
print('************************************')
print('Fitting is done!')

for name, par in result.params.items():
     print("  %s: value=%f +/- %f " % (name, par.value, par.stderr))
# valI1 = result.params['g1_area'].value
# print('instrument limit is {}'.format(ins_res(result.params['g1_mean'].value,3.32)))
# print(valI1)

fig2, ax2 = plt.subplots()
# x_e, sumII_q, err_q
ax2.plot(x_fit,y_fit-comps['bg_'],marker='.', ls='none', mfc='None', label='data')
# ax2.plot(x_fit,comps['bg_'],marker='.', ls='none', mfc='None', label='data')
ax2.plot(x_fit,result.best_fit-comps['bg_'],label='fit to data')
for i in range(num_of_peaks):
    ax2.plot(x_fit, comps['l{}_'.format(i+1)], label=r'Gaussian #{}'.format(i+1))
ax2.set_xlim(left=0.25,right=2.5)
# ax2.set_ylim(bottom=-0.01,top = 0.01)



# # para = add_uncertainty(pop, unc)
# # for i in range(3, len(pop), 3):
# #     hliney = 0.5 * pop[i] + exp_dec(pop[i + 1], pop[0], pop[1], pop[2])
# #     axi.hlines(hliney, xmin=pop[i + 1] - ins_res(pop[i + 1], 3.32) / 2, xmax=pop[i + 1] + ins_res(pop[i + 1], 3.32) / 2,
# #                color='r',
# #                label='instrument resolution')
# #     text += 'Gaussian peak{}, c = {}, fwhm = {}\n'.format(int(i / 3), para[i + 1], para[i + 2])
# # text += 'Instrument resolution:{}\n'.format(round(ins_res(0, 3.32), 3))
# #
# #
###****************************************************
###------------ fit neutron with parameters-------------
###****************************************************
# # print("-----------point charge model calculation --------------")
# x_e_long = np.linspace(0, 12, 1000)
# # Tb166LigP, TbP = point_charge()
# #
# #
# # B20P = 0.3392873
# # B40P = 0.00085596
# # B60P = 2e-08
# # B66P = 4.07e-06
# # #
# # BdictionaryP = {'B20': B20P, 'B40': B40P,
# #                'B60': B60P, 'B66': B66P}
# # TVS2 = cry_fie_cal(BdictionaryP)
# # #
# # print("-----------B factor  calculation --------------")

# # # adding B60 - case I
B20 = -0.132
B40 = -0.000508
B60 = 2.26021e-6
# B66List=[0]
B66List=[50e-6]

# # adding B60 - case II
# B20 = -0.111654
# B40 = -0.000541319
# B60 = 1.9127e-6
# B66List=[30e-6*1]

# # adding B60 - case I
# B20 = -0.104
# B40 = -0.000577
# B60 = 2.23e-6
# # B66List=[0]
# B66List=[2.5e-6]

# B66List=[2e-6*15]

fig3, ax3 = plt.subplots()
for B66 in B66List:
# B66=0
#
# B20 = -0.0666083
# B40 = -0.00078702
# B60 = 3.53771e-6
# B66 = 0
#
    Bdictionary = {'B20': B20, 'B40': B40,
                   'B60': B60, 'B66': B66}
    # #
    TVS = cry_fie_cal(Bdictionary)
    xx = np.linspace(0,12,1000)

    sim_int_40 = 0.0005 * TVS.normalizedNeutronSpectrum(xx, Temp=30, ResFunc=ins_res)
    ax3.plot(xx, sim_int_40, label='parameter1_40K_B66{}'.format(B66), marker='.',markersize=2)
    # ax3.set_xlim(left=0.25,right=2)
    # ax3.set_ylim(bottom=-1e-4,top=5e-3)

TVS.printLaTexEigenvectors()

plt.show()
