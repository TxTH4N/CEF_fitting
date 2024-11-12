import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
from scipy.optimize import curve_fit


# data = np.genfromtxt('data.txt', skip_header=1, unpack=True)
# tem = data[0]
# S = data[1]
# cmag = data[2]

# fig1, ax1 = plt.subplots()
# # ax1.plot(tem,S,marker='.',markersize=2)
# ax1.plot(np.log10(tem),S,marker='.',markersize=2)
# ax1.set_xlabel("$log_{10}T$")
# ax1.set_ylabel("S/R")
# ax1.hlines(np.log(2),xmin=0,xmax=2,linestyles='--',color='red')
# ax1.hlines(np.log(13),xmin=0,xmax=2,linestyles='--',color='green')
# ax1.set_xlim(left=0,right=2)
# ax1.set_ylim(bottom=0,top =3)

# fig2, ax2 = plt.subplots()
# ax2.plot(tem,S,marker='.',markersize=2)
# ax2.set_xlabel("T$(K)$")
# ax2.set_ylabel("S/R")
# ax2.hlines(np.log(2),xmin=min(tem),xmax=max(tem),linestyles='--',color='red')
# ax2.set_xlim(left=0,right =10)
# ax2.set_ylim(bottom=0,top =np.log(2)+0.2)

# fig3, ax3 = plt.subplots()
# # ax1.plot(tem,S,marker='.',markersize=2)
# ax3.plot(tem,cmag,marker='.',markersize=2)
# ax3.set_xlabel("$T$ (K)")
# ax3.set_ylabel("$C_{p}$")

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
    fig, ax = plt.subplots(figsize = (4,3))
    cp = ax.pcolormesh(q, e, int, cmap='jet', vmin=int_ran[0], vmax=int_ran[1])
    fig.colorbar(mappable=cp)
    ax.vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    ax.vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    ax.hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    ax.hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    # fig.show()
    return fig, ax

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

path = '../data/ASCII_file/'
# q_range = [1.42, 1.55]  ## q bin range
# q_range = [1.3, 1.42]
q_range = [1.3, 1.42]
e_range = [-0.15, 0.15]  ## plot range
Ei='3p32'
# bg_10K = 'Ei{}_T{}.iexy'.format(Ei, 10)
# q_bg, e_bg, data_bg, mask_i_bg, mask_di_bg = load_data(path, bg_10K)
# sumII, err = int(q_bg, e_bg, mask_i_bg, mask_di_bg, q_range, e_range)
# print(sumII)

file_list=['1p7',3,4,5,6,8,10,20]
tem = [1.7,3,4,5,6,8,10,20]
ii_list=[]
err_list=[]
for idx, T in enumerate(file_list):
    file = 'Ei{}_T{}.iexy'.format('3p32', T)
    q, e, data, mask_i, mask_di = load_data(path, file)
    sumII, err = int(q, e, mask_i, mask_di, q_range, e_range)
    ii_list.append(sumII)
    err_list.append(err)
fig,ax = plt.subplots()
ax.errorbar(tem, ii_list, err_list, marker='.', ls='none', fmt='o-', mfc='None', label='data')
ax.set_xlabel('T (K)',fontsize=15)
ax.set_ylabel('II (a.u.)',fontsize=15)
ax.set_title('Integrated intensity on (1 0 1)',fontsize=15)
ax.tick_params(axis='both', labelsize=13)

# plt.savefig("order parameter (1 0 1).png",dpi=300)
plt.savefig("order parameter (1 0 0).png",dpi=300)




# data = data-data_bg
# mask_i = mask_i-mask_i_bg
# mask_di = mask_di-mask_di_bg
config, conax = contourPlot(q, e, mask_i, [0,10])
plt.show()

print(ii_list)

data = np.column_stack([tem,ii_list,err_list])
datafile_path = "/your/data/output/directory/datafile.txt"
np.savetxt('intensity.txt', data,fmt='%.2f',header='Temperature    II   uncertainty')
