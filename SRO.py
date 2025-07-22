import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
from lmfit import Model, Parameters
import scipy.special as spf
import matplotlib
import load_data as ld
import timeit

matplotlib.use('QtAgg')


def gaussian(x, amplitude, mean, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    y = amplitude / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mean) ** 2 / 2 / (sigma ** 2))
    return y


def lorentzian(x, amplitude, mean, gamma):
    y = amplitude / np.pi * 0.5 * gamma / ((x - mean) ** 2 + (0.5 * gamma) ** 2)
    return y


def voigt(x, amplitude, mean, fwhm, gamma):
    hwhm = gamma / 2
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    voi = amplitude * spf.voigt_profile(x - mean, sigma, hwhm)
    return voi


def linear_bg(x, slope, intercept):
    y = slope * x + intercept
    return y


def exp_dec_bg(x, a, beta, c):
    y = a * np.exp(-x * beta) + c
    return y


def lorentzian_bg(x, amplitude, mean, gamma, slope, intercept):
    y = amplitude / np.pi * 0.5 * gamma / ((x - mean) ** 2 + (0.5 * gamma) ** 2) + slope * x + intercept
    return y


def ins_res_3p32(x):
    y = 0.00030354 * x ** 3 + 0.0039009 * x ** 2 - 0.040862 * x + 0.11303
    return y


def ins_res_12(x):
    y = 3.8775e-05 * x ** 3 + 0.0018964 * x ** 2 - 0.078275 * x + 0.72305
    return y


def ins_res_1p55(x):
    y = +0.00096336 * x ** 3 + 0.005865 * x ** 2 - 0.027999 * x + 0.038397
    return y


# def disp(q, e, ii, di, Q_range, E_range,sumE=False):
#     q_ind = np.where(np.logical_and(q >= Q_range[0], q <= Q_range[1]))[0]  ## get q bin index
#     e_ind = np.where(np.logical_and(e >= E_range[0], e <= E_range[1]))[0]
#     ex = np.array([ei for ei in e[e_ind]])
#     qx = np.array([qi for qi in q[q_ind]])
#     IIMat = ii[e_ind, :][:, q_ind]
#     errMat = di[e_ind, :][:, q_ind]
#     if sumE is False:
#         sumII = np.nanmean(IIMat, axis=1)
#         err_sq = np.nansum(np.square(errMat), axis=1)
#         n = np.sum(1 - np.isnan(errMat), axis=1)
#         err = np.sqrt(err_sq) / n
#         return ex, sumII, err
#     elif sumE is True:
#         sumII = np.nanmean(IIMat, axis=0)
#         err_sq = np.nansum(np.square(errMat), axis=0)
#         n = np.sum(1 - np.isnan(errMat), axis=0)
#         err = np.sqrt(err_sq) / n
#         return qx, sumII, err


def spectrum(obj, Earray, Temp, ResFunc, gamma=0):
    eigenkets = obj.eigenvectors.real
    intensity = np.zeros(len(Earray))
    beta = 1 / (8.61733e-2 * Temp)
    Z = sum([np.exp(-beta * en) for en in obj.eigenvalues])
    for i, ket_i in enumerate(eigenkets):
        pn = np.exp(-beta * obj.eigenvalues[i]) / Z
        if pn > 1e-3:  # only compute for transitions with enough weight
            for j, ket_j in enumerate(eigenkets):
                # compute amplitude
                # mJn = self._transition(ket_i,ket_j)  # Old: slow
                mJn = obj.opttran.transition(ket_i, ket_j)
                deltaE = obj.eigenvalues[j] - obj.eigenvalues[i]
                if deltaE > 0.2:
                    GausWidth = ResFunc(deltaE)  # peak width due to instrument resolution
                    intensity += ((pn * mJn * obj._voigt(x=Earray, x0=deltaE, alpha=GausWidth,
                                                         gamma=gamma)).real).astype('float64')
    return intensity


def contour_12(q, e, mask_i, energyRange):
    config, conax = plt.subplots()
    cp = conax.pcolormesh(q, e, mask_i, cmap='jet', vmin=energyRange[0], vmax=energyRange[1])
    config.colorbar(mappable=cp)
    conax.set_xlabel("Q $(Å^{-1})$")
    conax.set_ylabel("$\\hbar \\omega$ (meV)")
    # conax.hlines(3, xmin=(min(q)), xmax=(max(q)), ls='--', color='yellow')
    conax.set_ylim(bottom=-0.5)
    conax.set_xlim(right=4.5)
    return config, conax


def contour_3(q, e, mask_i, energyRange):
    config, conax = plt.subplots()
    cp = conax.pcolormesh(q, e, mask_i, cmap='jet', vmin=energyRange[0], vmax=energyRange[1])
    config.colorbar(mappable=cp)
    conax.set_xlabel("Q $(Å^{-1})$")
    conax.set_ylabel("$\\hbar \\omega$ (meV)")
    # conax.set_ylim(bottom=-0.1)
    conax.set_ylim(bottom=-1)
    conax.set_xlim(right=2.3)
    # fig.show()
    return config, conax


def contour_1(q, e, mask_i, energyRange):
    config, conax = plt.subplots()
    cp = conax.pcolormesh(q, e, mask_i, cmap='jet', vmin=energyRange[0], vmax=energyRange[1])
    config.colorbar(mappable=cp)
    conax.set_xlabel("Q $(Å^{-1})$")
    conax.set_ylabel("$\\hbar \\omega$ (meV)")
    conax.set_ylim(bottom=-0.1)
    conax.set_xlim(right=1.6)
    # fig.show()
    return config, conax


ion = 'Tb3+'
folder = '/Users/tianxionghan/research/CEF_cal/CrystalFieldCal/data/measurement_2/'
ei_list = [3.32]
t_list = [6, 10,20, 30, 40, 60]
# t_list = [12.5]
field_list = []
data_list = []
start = timeit.default_timer()
for temp in t_list:
    for ei in ei_list:
        if field_list == []:
            data_list.append(ld.data(folder, ei, temp, field=None, specialCon='old_'))
        else:
            for ti in field_list:
                data_list.append(ld.data(folder, ei, temp, field=ti, specialCon='powder_'))
stop = timeit.default_timer()
print('Inport data sets in {} seconds '.format(stop - start))

# energeRange = [0, 0.005]
energeRange = [0, 1]

q_range = [0.25, 2.2]  ##3.32 meV
e_range = [-0.2, 0.2]

# q_range = [0.5, 1.1]
# e_range = [0.29, 2.4]  ## plot range

fit_range = [0.12, 2.6]

num_of_peaks = 6
fitted_results_list = []
bg = np.array([1.4023718117324189, 1.2364672473512694, 1.1238775265030874, 1.0472633011337518, 0.9325454500123062, 0.8433255814659845, 0.7936514741072872, 0.7784910504071364, 0.7842564561816223, 0.7856407089877541, 0.7955615231017248, 0.7688798420809082, 0.75790291652375, 0.7427463907314381, 0.7321171517122091, 0.7332415723924944, 0.7194716714555959, 0.7168904289876626, 0.7139266723271144, 0.7146164106152518, 0.7096499780799502, 0.7172912988821921, 0.728657701806141, 0.7122770276559683, 0.6879883362870934, 0.690052591153517, 0.6850861470374204, 0.6901502023646557, 0.6860376628773648, 0.6879006978361478, 0.6779104596565139, 0.6717832898213062, 0.6687110189268731, 0.676338857945456, 0.68619158768543, 0.6749871873824128, 0.6735159780949177, 0.6709209160794679, 0.6627334909383416, 0.6619177061254528, 0.6642931869622404, 0.6497754255016149, 0.6245144374420061, 0.6431283320136468, 0.6672273609433198, 0.6565848588189486, 0.6484307877050434, 0.6407610630111371, 0.6621281963117484, 0.6512507256741861, 0.6863652354784021, 0.7234801735640097, 0.8627270638577315, 1.1614394860614374, 1.1766271215056143, 1.6606968014567418, 1.6930687945841285, 1.2039866586235368, 0.7597660076151899, 0.6309415968153425, 0.6284857811427794, 0.6390755916133872, 0.6515348183400697, 0.638763788301555, 0.6344545464939993, 0.6018732198747786, 0.6011186448364064, 0.5892160802432942, 0.6215823614243076, 0.6148969099566032, 0.5981799749497516, 0.5797309021677118, 0.5829818900305308, 0.5678547043145283, 0.6069749167771238, 0.6056397198489943, 0.6062163671445129, 0.6531462346900242, 0.6247852368984216, 0.5953340878138974, 0.7648849464305028, 0.9373724377720345, 1.309147945315842, 0.9577732098140315, 0.6031656460493369, 0.6084625620747396, 0.5857706222514367, 0.5689144009354831, 0.6315146648533149, 0.7496245822193391, 0.853351983591767, 0.6645369104255917, 0.571782528305238, 0.5652664194636469, 0.7137702525783617, 1.0586677189450027, 0.7885004351203749
])
bg_er = np.array(
    [0.00624017167727219, 0.004823197840866039, 0.004133714867066775, 0.004475522300355098, 0.002970195554373565, 0.0023926636685396283, 0.002335133382297825, 0.002340984802362928, 0.0024208381411821747, 0.002716274203375896, 0.002443544757310893, 0.002209905769454914, 0.002208494301171922, 0.002181384864654473, 0.0022745193570948515, 0.002429615284143131, 0.002323200399307874, 0.002393379453690127, 0.0021538124232428945, 0.002124079094194238, 0.002117671431127344, 0.0022205438458551686, 0.0022220948203814455, 0.0023702432791787038, 0.0022280689969429173, 0.002207830258365436, 0.0020723618973920266, 0.0021782350892807626, 0.00205775027984358, 0.0020229941627571287, 0.0019600096834001103, 0.0020386096820045494, 0.001917512795412134, 0.0019171382257744472, 0.002105783304411718, 0.0019205031238532832, 0.0018913274188931612, 0.002030856181173963, 0.0019962453907274953, 0.0019872375800285167, 0.0020080411691055945, 0.0019243615890653818, 0.0018943363962463076, 0.0019088766823093983, 0.0019128568413374078, 0.001908124166794527, 0.001947777614259909, 0.0019209223345358817, 0.001857083571459272, 0.0018349833727514685, 0.0018884186439068692, 0.0020558721379122106, 0.0021915149486675073, 0.002487738987546492, 0.0025348888995251483, 0.0029973783580354435, 0.002861983643780368, 0.0024730205119257356, 0.0020396260855499006, 0.001862791798033563, 0.0018343351677131408, 0.0017380198671939006, 0.0018197754568986862, 0.0017684762737670936, 0.0018381048726685868, 0.0017605079247289348, 0.0016209446545908477, 0.0017942685303922962, 0.0020720265601152157, 0.001983765864154173, 0.0019709393929592564, 0.0018882151980135298, 0.0016986348407440311, 0.0017811589437231148, 0.002030638057486805, 0.0019432729673190673, 0.0019510279408970273, 0.001743506758073994, 0.0017521767808964313, 0.0016121406955154464, 0.0018861050765525455, 0.0021690476541759035, 0.002321692544670064, 0.0019573494173008204, 0.001724748193249856, 0.00155566687043957, 0.001642298787350993, 0.001497703257216469, 0.0015293024024290115, 0.0016512344621935817, 0.0016804888019937307, 0.0014427671316430525, 0.0013557529647937978, 0.0013394852449320233, 0.0014874798751454683, 0.0018857818272274651, 0.001550777972059397
])

# fig, ax = plt.subplots()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 8),gridspec_kw={'hspace': 0})
for idx, measurement in enumerate(data_list):
    q = measurement.q
    e = measurement.e
    mask_i = measurement.mI
    mask_di = measurement.mdI
    config, conax = contour_3(q, e, mask_i, energeRange)
    conax.vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    conax.vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    conax.hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    conax.hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')

    # Calculate dispersion spectrum below
    measurement.disp(q_range, e_range, sumE=True)

    x_q = measurement.qx
    intensity = measurement.sumII
    print('nan' in intensity)
    error = measurement.err
    if measurement.temp ==60:
        continue
    ax1.errorbar(x_q, intensity - bg, yerr=np.sqrt(error**2+bg_er**2), fmt='.-', mfc='None', color='C{}'.format(idx),
                label='{} K - 60 K'.format(measurement.temp))
    # ax.set_xlabel('$\\hbar \\omega$ (meV)',fontsize=20)
    # ax.set_ylabel('$\\rm I$ (a.u.)',fontsize=20)
ax1.set_ylabel('Intensity', fontsize=20)
# ax.set_title('CEF e',fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_xlim(left=0.3, right=2.2)
ax1.set_ylim(bottom=0, top=0.2)

ax1.legend(loc='upper left',fontsize=12,bbox_to_anchor=(0.1, 1))
ax1.tick_params(top=True, bottom=True, left=True, right=True,  # turn on all ticks
               labeltop=False, labelbottom=True, labelleft=True, labelright=False)  # optional: show labels
ax1.tick_params(direction='in')  # 'in', 'out', or 'inout'
# plt.show()
# formatted_bg = ", ".join(map(str, intensity))
# formatted_err = ", ".join(map(str, error))
# print(formatted_bg)
# print(formatted_err)

def form_factor(q):
    s =np.abs(q)/4/np.pi
    g = 3/2
    FFj0A= 0.0177
    FFj0a=25.5095
    FFj0B= 0.2921
    FFj0b=10.5769
    FFj0C= 0.7133
    FFj0c= 3.5122
    FFj0D=-0.0231
    FFj2A= 0.2892
    FFj2a=18.4973
    FFj2B= 1.1678
    FFj2b= 6.7972
    FFj2C= 0.9437
    FFj2c= 2.2573
    FFj2D= 0.0232
    j0 = FFj0A*np.exp(-FFj0a*s**2)+FFj0B*np.exp(-FFj0b*s**2)+FFj0C*np.exp(-FFj0c*s**2)+FFj0D
    j2 = (FFj2A*np.exp(-FFj2a*s**2)+FFj2B*np.exp(-FFj2b*s**2)+FFj2C*np.exp(-FFj2c*s**2)+FFj2D)*s**2
    return j0+(2-g)/g*j2
# Q = np.linspace(0.01, 2.5, 300)
a = 5.53
c = 9.023
neighbors = np.array([
    [ a, 0,0],
    [-a, 0,0],
    [ a/2,  a*np.sqrt(3)/2,0],
    [-a/2,  a*np.sqrt(3)/2,0],
    [ a/2, -a*np.sqrt(3)/2,0],
    [-a/2, -a*np.sqrt(3)/2,0],
])

r_vectors = np.vstack(([[0, 0,0]], neighbors))  # central spin + neighbors
print(r_vectors)
N = len(r_vectors)

# Powder average setup
Q_vals = np.linspace(0, 2.5, 300)
Q_vals = np.linspace(0, 6, 300)
n_theta = 60
n_phi = 60
theta_list = np.linspace(0, np.pi, n_theta)
phi_list=np.linspace(0, 2*np.pi, n_phi)
S_Q = []

# Compute powder-averaged structure factor
for Q in Q_vals:
    print(Q)
    Q_sum = 0
    for theta in theta_list:
        for phi in phi_list:
        # q_vec = Q * np.array([np.cos(theta), np.sin(theta)])
            q_vec = Q * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            S_q = 0
            for i in range(N):
                for j in range(N):
                    r_diff = r_vectors[i] - r_vectors[j]
                    S_q += np.cos(np.dot(q_vec, r_diff))*(1-np.cos(theta)**2)

                    # S_q += np.exp(1j*(np.dot(q_vec,r_diff)))
                    # S_q_sq =np.real(S_q)
            Q_sum += S_q*np.sin(theta)/N**2
    S_Q.append(Q_sum / (n_theta*n_phi))

S_Q = np.array(S_Q)
I_Q = form_factor(Q_vals)**2 * S_Q
print(I_Q)
# Plot result
# plt.figure(figsize=(7, 4))
# plt.plot(Q_vals, I_Q, label='Hexagonal FM cluster $S(Q)$')
# plt.xlabel(r'$Q$ ($\mathrm{\AA}^{-1}$)')
# plt.ylabel('Scattering Intensity (a.u.)')
# plt.title('Powder-Averaged Structure Factor for Hexagonal Tb$^{3+}$ Cluster')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# FM dimer structure factor
# FQ2 = form_factor(Q)**2
# interference = 1 + np.sin(Q * d) / (Q * d)
# S_Q = FQ2 * interference
ax2.plot(Q_vals, I_Q, label='NN Tb$^{3+}$ dimmer Structure factorR')
ax2.set_xlabel(r'$Q (\AA^{-1})$', fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.set_ylabel('Powder-averaged', fontsize=20)

ax2.set_ylim(bottom=0, top=0.135)
# ax2.set_ylim(bottom=0, top=3)
ax2.set_xlim(left=0.2, right=2.1)

ax2.tick_params(top=True, bottom=True, left=True, right=True,  # turn on all ticks
               labeltop=False, labelbottom=True, labelleft=True, labelright=False)  # optional: show labels
ax2.text(0.75,0.32,'Hexagonal Tb-Tb spin cluster',fontsize=16)
ax1.text(0.35,0.015,'(a)',fontsize=20)
ax2.text(0.35,0.32,'(b)',fontsize=20)
# Optionally: set tick direction
ax2.tick_params(direction='in')  # 'in', 'out', or 'inout'
# ax.set_yticks(np.linspace(-1, 1, 4))
fig.tight_layout()
plt.show()


# from scipy.io import loadmat
# data = loadmat('triangular_cluster_intensity.m')
# labels=list(data.keys())
# print(labels)