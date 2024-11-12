import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
from scipy.optimize import curve_fit


class data:
    def __init__(self, folder, ei, temperature):
        self.ei = ei
        self.temp = temperature
        self.name = '{}meV_{}K'.format(ei,temperature)
        temp_str = str(temperature)
        ei_str = str(ei)
        if '.' in temp_str:
            temp_str = temp_str.replace('.', 'p')
        if '.' in ei_str:
            ei_str = ei_str.replace('.', 'p')
        filename = "Ei{}_T{}.iexy".format(ei_str, temp_str)
        self.path = folder + filename
        self.rawdata = np.genfromtxt(self.path, skip_header=1, unpack=True)
        self.q = np.unique(self.rawdata[2])
        self.e = np.unique(self.rawdata[3])
        n = len(self.e)
        mdata = np.ma.masked_where(self.rawdata[0] == -1e+20, self.rawdata[0])
        mdI = np.ma.masked_where(self.rawdata[1] == -1, self.rawdata[1])
        mii = np.transpose(np.array([mdata[i:i + n] for i in range(0, len(mdata), n)]))
        mdI = np.transpose(np.array([mdI[i:i + n] for i in range(0, len(mdI), n)]))
        self.mI = mii.copy()
        self.mI[mii < -1e5] = None
        self.mdI = mdI.copy()
        self.mdI[mdI < 0] = None
        print('Read data in the shape of {} in e and {} in q'.format(len(self.e), len(self.q)))

    # def cplot(self, int_range, q_range, e_range):
    #     q = self.q
    #     e = self.e
    #     int = self.mii
    #     fig, ax = plt.subplots()
    #     cp = ax.pcolormesh(q, e, int, cmap='jet', vmin=int_range[0], vmax=int_range[1])
    #     cbar = fig.colorbar(mappable=cp)
    #     ax.set_xlabel("Q $(Å^{-1})$", size=25)
    #     ax.set_ylabel("$\\hbar \\omega$ (meV)", size=25)
    #     ax.vlines(min(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    #     ax.vlines(max(q_range), ymin=min(e_range), ymax=max(e_range), ls='--', color='r')
    #     ax.hlines(min(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    #     ax.hlines(max(e_range), xmin=(min(q_range)), xmax=(max(q_range)), ls='--', color='r')
    #     ax.set_xticks(np.arange(0.5, 4.6, 1))
    #     ax.set_yticks(np.arange(-2, 13, 2))
    #     ax.tick_params(axis='both', labelsize=25)
    #     cbar.ax.tick_params(labelsize=25)
    #     ax.set_ylim(bottom=-0.5)
    #     ax.set_xlim(right=4.5)
    #     fig.tight_layout()
    #     return fig, ax


def main():
    import timeit
    # start = timeit.default_timer()

    folder = '/Users/tianxionghan/research/CrystalFieldCal/data/ASCII_file/'
    # data1 = data(folder, 1.7, 3.32)
    data1 = data(folder, 12, 40)
    print(data1.__dir__())
    intensity = [0, 0.01]
    q_range = [1.2, 2.4]
    e_range = [3, 10]


if __name__ == "__main__":
    main()