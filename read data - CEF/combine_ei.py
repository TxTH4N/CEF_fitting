import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
from scipy.optimize import curve_fit
import scipy.special as spf
import load_data as ld
import timeit
import matplotlib
matplotlib.use('QtAgg')
from mpl_toolkits.axes_grid1 import make_axes_locatable

def contourPlot(obj, int_ran,q_range,e_range):
    fig, ax = plt.subplots(figsize=(3,2.25))
    cp = ax.pcolormesh(obj.q, obj.e, obj.mI, cmap='jet', vmin=int_ran[0], vmax=int_ran[1])
    cbar =fig.colorbar(mappable=cp,ticks=[0,0.005,0.01])
    cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.tick_params(labelsize=12)
    # ax.set_title("cplot_{}K_Ei{}".format(obj.temp, obj.ei))
    ax.set_xlabel("Q $(Å^{-1})$",fontsize=14)
    # ax.set_ylabel("$\\hbar \\omega$ (meV)",fontsize=10)
    ax.set_ylabel('$\\Delta E$ (meV)', fontsize=14)
    ax.vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    ax.vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    ax.hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    ax.hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    # fig.tight_layout()
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

# ####--------------------------------------------
folder = '../data/ASCII_file/'
# ei_list = [12,3.32]
# t_list= [10,30,40,60]
ei_list = [12,3.32]
t_list= [10,30,40,60]
data_list_12 = []
data_list_3p32 = []

start = timeit.default_timer()

for temp in t_list:
    for ei in ei_list:
        if ei ==12:
            data_list_12.append(ld.data(folder, ei, temp))
        else:
            data_list_3p32.append(ld.data(folder, ei, temp))
stop = timeit.default_timer()
print('Inport data sets in {} seconds '.format(stop-start)) ### 0-4 3.32meV, 5-9 12meV

#
Int_Range = [0, 1]
# old plot
# q_range_12 = [1.2, 2.4]  ## q bin range
# e_range_12 = [3, 11]  ## plot range
# q_range_3p32 = [0.6, 1.8]  ## q bin range
# e_range_3p32 = [0.29, 2.9]  ## plot range
#
# new plot
q_range_12 = [1.3,1.9]  ## q bin range
e_range_12 = [3, 10]  ## plot range
q_range_3p32 = [1.3,1.9]  ## q bin range
e_range_3p32 = [0.2,2.7]  ## plot range
#
#
data = data_list_12[1]
data2 = data_list_3p32[1]
data_list = [data, data2]
q_range=[q_range_12, q_range_3p32]
e_range=[e_range_12, e_range_3p32]
# data.config, data.conax = contourPlot(data, Int_Range,q_range_12,e_range_12)
fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(4.5, 6),layout='compressed')
for idx,ax in enumerate(axes):
    im = ax.pcolormesh(data_list[idx].q,data_list[idx].e,data_list[idx].mI*100,cmap='jet', vmin=Int_Range[0], vmax=Int_Range[1])
    ax.vlines(min(q_range[idx]), ymin=min(e_range[idx]), ymax=max(e_range[idx]), ls='--', color='r')
    ax.vlines(max(q_range[idx]), ymin=min(e_range[idx]), ymax=max(e_range[idx]), ls='--', color='r')
    ax.hlines(min(e_range[idx]), xmin=(min(q_range[idx])), xmax=(max(q_range[idx])), ls='--', color='r')
    ax.hlines(max(e_range[idx]), xmin=(min(q_range[idx])), xmax=(max(q_range[idx])), ls='--', color='r')
    ax.tick_params(axis='both', labelsize=20)
# axes[0].set_xlabel("Q $(Å^{-1})$", fontsize=14)
axes[0].set_ylabel('$\\Delta E$ (meV)', fontsize=20,labelpad=10)
axes[0].hlines(3, xmin=(min(data_list[0].q)), xmax=(max(data_list[0].q)), ls='--', color='yellow')
axes[0].set_xticks([0.5,2.5,4.5])
axes[0].set_yticks([0,3,6,9])
axes[0].set_ylim(bottom=-0.5,top=11)
axes[0].set_xlim(left = 0.5,right=4.5)


axes[1].set_xticks([0.25,1.25,2.25])
axes[1].set_yticks([0,1,2,3])
axes[1].set_ylim(bottom=-0.1,top=3)
axes[1].set_xlim(left = 0.25,right=2.25)
axes[1].set_ylabel('$\\Delta E$ (meV)', fontsize=20,labelpad=10)
axes[1].set_xlabel("Q $(Å^{-1})$", fontsize=20)
# axes[1].vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
# axes[1].vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
# axes[1].hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
# axes[1].hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
cbar = fig.colorbar(mappable=im,ax=axes.ravel().tolist(), ticks=[0,1],pad=0.01,shrink=1)
# cbar.formatter.set_powerlimits((0, 0))
cbar.set_ticklabels(['0', 'MAX'],fontsize=20)

# plt.savefig("../plots/cont_both_ver",dpi=300)

#
# data2 = data_list_3p32[1]
# # data2.config, data2.conax = contourPlot(data2, Int_Range,q_range_3p32,e_range_3p32)
#
# fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(3, 2.25))
# cp = ax.pcolormesh(obj.q, obj.e, obj.mI, cmap='jet', vmin=int_ran[0], vmax=int_ran[1])
# cbar = fig.colorbar(mappable=cp, ticks=[0, 0.005, 0.01])
# cbar.formatter.set_powerlimits((0, 0))
# cbar.ax.tick_params(labelsize=12)
# # ax.set_title("cplot_{}K_Ei{}".format(obj.temp, obj.ei))
# ax.set_xlabel("Q $(Å^{-1})$", fontsize=14)
# # ax.set_ylabel("$\\hbar \\omega$ (meV)",fontsize=10)

#

# data2.conax.set_title("T = 1.7K, Ei = 3.32meV")
# data2.conax.set_ylim(bottom=-0.1,top=3)
# data2.conax.set_xlim(left = 0.25,right=2.25)
# # data2.conax.tick_params(axis='both', labelsize=12)
# data2.config.tight_layout(pad = 0.04)
# plt.show()
# plt.savefig("../plots/cont_3p32meV_30K",dpi=300)
# plt.savefig("../plots/cont_3p32meV_1p7K",dpi=300)

#
# fig1, ax1 = plt.subplots(figsize=(3,2))
# # fig1, ax1 = plt.subplots(2,5,figsize=(6.75,3.4),gridspec_kw={'width_ratios': [5,5,5,5,1.5]})
# for idx,data in enumerate(data_list_12):
#     data.x_e, data.sumII_q, data.err_q = disp(data.q, data.e, data.mI, data.mdI, q_range_12, e_range_12)
#     ax1.errorbar(data.x_e, data.sumII_q, data.err_q, marker='.', ls='-',linewidth=0.7,markersize=1.5,
#                  label="{} K".format(data.temp))
#     # ax1.set_xlabel('$\\hbar \\omega$ (meV)',fontsize=10)
#     ax1.set_xlabel('$\\hbar \\omega$ (meV)', fontsize=12)
#     ax1.set_ylabel('$\\rm I$ (a.u.)',fontsize=12)
#     ax1.tick_params(axis='both', labelsize=12)
#     ax1.set_yticks([0,0.01,0.02,0.03,0.035])
#     ax1.set_xticks([3,4,5,6,7,8,9])
#     ax1.set_xlim(left=3, right=9)
#     ax1.set_ylim(bottom=0, top=0.035)
# ax1.legend(fontsize=10)
# fig1.tight_layout(pad = 0.05)
# plt.show()
#
# # plt.savefig("../plots/dispersion_12meV",dpi=300)
#
#
#
# fig2, ax2 = plt.subplots(figsize=(3,2))
# for idx,data in enumerate(data_list_3p32):
#     data.x_e, data.sumII_q, data.err_q = disp(data.q, data.e, data.mI, data.mdI, q_range_3p32, e_range_3p32)
#     ax2.errorbar(data.x_e, data.sumII_q, data.err_q, marker='.', ls='-', linewidth=0.7, markersize=1.5,
#                  label="{} K".format(data.temp))
#     # ax2.set_xlabel('$\\hbar \\omega$ (meV)', fontsize=10)
#     ax2.set_xlabel('$\\Delta E$ (meV)', fontsize=14)
#     ax2.set_ylabel('$\\rm I$ (a.u.)', fontsize=14)
#     ax2.tick_params(axis='both', labelsize=14)
#     ax2.set_yticks([0, 0.005, 0.01, 0.015, 0.018])
#     ax2.set_xticks([0.2,1,2,3])
#     ax2.set_xlim(left=0.2, right=3)
#     ax2.set_ylim(bottom=0, top=0.018)
# ax2.legend(fontsize=8)

# # plt.savefig("../plots/dispersion_3p32meV",dpi=300)

plt.show()
