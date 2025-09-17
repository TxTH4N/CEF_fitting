import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model, Parameters
from matplotlib.ticker import MultipleLocator

def path(instrument, exp_num, scan_num):
    path = os.path.abspath("/Users/tianxionghan/research/TbV6Sn6/IPTS-32293_TbV6Sn6/IPTS-32293")
    file = path + '/exp{}/Datafiles/{}_exp0{}_scan0'.format(exp_num, instrument, exp_num) + str(scan_num) + ".dat"
    return file


def get_info(instrument, exp_num,scanNum):

    list = []
    label = {'samplename': [], 'lattice constant': [], 'scan': [], 'x': [], 'y': [], 'ylabel': [],
             'temperature': [], 'tem_error': []}
    x = []
    y = []
    temperature = []
    yerr = []

    monitor=[]
    # with open(file,"r", encoding='utf-8') as f:         # Open file
    with open(path(instrument, exp_num,scanNum), "r") as f:
        lines = f.readlines()  # Read all the lines in the file

    for i in range(len(lines)):
        line = lines[i].split()
        if line[0] != '#':
            list.append(line)
        elif line[1] == 'latticeconstants':
            label['lattice constant'] = (line[3:])
        elif line[1] == 'def_x':
            label['x'] = line[-1]
        elif line[1] == 'def_y':
            label['y'] = line[-1]
        elif line[1] == 'samplename':
            label['samplename'] = line[-1]
        elif line[1] == 'scan':
            label['scan'] = line[-1]
        elif line[1] == 'Pt.':
            x_num = line.index(label['x']) - 1
            y_num = line.index(label['y']) - 1
            time_ind = line.index('time') - 1
            monitor_ind = line.index('monitor') - 1
            t_sample_ind = line.index('dr_tsample') - 1
    for i in range(len(list)):
        x.append(float(list[i][x_num]))
        monitor.append(float(list[i][monitor_ind]))
        temperature.append(float(list[i][t_sample_ind]))
    avg_monitor = np.average(monitor)
    avg_tem=np.average(temperature)
    label['ylabel'] = 'Intensity (counts per monitor)'
    for i in range(len(list)):
        time = float(list[i][time_ind])
        y.append(float(list[i][y_num])/monitor[i] * (avg_monitor /time))
        yerr.append(np.sqrt(float(list[i][y_num])) / monitor[i] * (avg_monitor / time))
    avg_tem = round(np.average(temperature), 2)
    tem_error = round(np.std(temperature) / np.sqrt(np.size(temperature)), 3)

    label['temperature'] = avg_tem
    label['tem_error'] = tem_error
    print("read data successfully with {} lines, with sample temperature {}".format(len(list), avg_tem)
          + '\u00B1' + '{}K'.format(tem_error))

    return label, np.array(x), np.array(y), list, yerr

def gaussian(x, amplitude, mean, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    y = amplitude * np.exp(-(x - mean) ** 2 / 2 / (sigma ** 2))
    return y
def multi_gaussian(x, *params):
    # k = params[0]
    # c = params[1]
    # y = k*x+c
    y = params[0]
    for i in range(1, len(params), 3):
        amplitude = params[i]
        mean = params[i + 1]
        fwhm = params[i + 2]
        y += gaussian(x, amplitude, mean, fwhm)
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

instrument = "CG4C"
exp_num = "416"
# scan = '023'
scanlist_1=['042','043','044','045','046','047','048','049','050','051','052']
scanlist_2=['053','054','055','056','057','058','059','060','061','062','063','064','065','067']
# scanlist=['042','043','044','045','046','047','049','050','051','052','053','054','055','056','057','058','059','060','061','062','063','064','065','067']

y_list=[]
y_err_list=[]
int = []
center = []
FWHM = []
temp = []
unc_I = []
labels= []
fig, ax = plt.subplots(2,1,figsize=(4.5,6))
for idx, scans in enumerate(scanlist_1+scanlist_2):
    # print(idx,scans)
    # fig_i,ax_i = plt.subplots()
    labels.append('C1' if scans in scanlist_1 else 'C0')
    label, x, y, lis, yerr = get_info(instrument, exp_num,scans)
    y_list.append(y)
    y_err_list.append(yerr)
    p0=[
        2,
        4, 70, 1
        ]
    if scans>'047' and scans<='056' :
        int.append(0)
        unc_I.append(0.01)
        temp.append(label['temperature'])
        ax[0].errorbar(x, y_list[idx], yerr=y_err_list[idx], label="{} K".format(temp[idx]), marker='.',
                       linestyle='None', color="C{}".format(idx))
        # ax_i.errorbar(x, y_list[idx], yerr=y_err_list[idx], label="{} K".format(temp[idx]), marker='.',
        #                linestyle='None', color="C{}".format(idx))
        continue
    pop, pco = curve_fit(multi_gaussian, x, y_list[idx], p0=p0)
    unc = np.sqrt(np.diag(pco))
    if pop[1]<0:
        pop[1] = 0
    int.append(pop[1])
    center.append(pop[2])
    FWHM.append(pop[3])
    unc_I.append(unc[1])
    temp.append(label['temperature'])

    # ax_i.errorbar(x, y_list[idx],yerr=y_err_list[idx], label="{} K".format(temp[idx]),marker='.', linestyle='None',color="C{}".format(idx))
    # ax_i.plot(x,multi_gaussian(x, *pop),color="C{}".format(idx),linestyle='--')
    # ax_i.set_xlabel(r'2$\theta$',fontsize=14)
    # ax_i.set_ylabel('Intensity (cts/sec)', fontsize=14)
    # ax_i.set_xlim(left=66,right=73)
    # ax_i.set_ylim(bottom=0, top=12)
    # ax_i.text(66.2, 11, "Bragg peak (1 0 1)", fontsize=10)

    ax[0].errorbar(x, y_list[idx],yerr=y_err_list[idx], label="{} K".format(temp[idx]),marker='.', linestyle='None',color="C{}".format(idx))
    ax[0].plot(x,multi_gaussian(x, *pop),color="C{}".format(idx),linestyle='--')
    ax[0].set_xlabel(r'2$\theta$',fontsize=14)
    ax[0].set_ylabel('Intensity (cts/sec)', fontsize=14)
    ax[0].set_xlim(left=66,right=73)
    ax[0].set_ylim(bottom=0, top=12)
    ax[0].text(66.2, 11, "Bragg peak (1 0 1)", fontsize=10)

temp = np.array(temp)
int = np.array(int)
print(', '.join(f'{v:.2f}' for v in temp))
# print(int)
print(', '.join(f'{v:.2f}' for v in int))
print(', '.join(f'{v:.2f}' for v in unc_I))
# fig, ax = plt.subplots()
for idx,(xi, yi, ei, ci) in enumerate(zip(temp, int, unc_I, labels)):
    if idx ==0:
    # if ci =='C1':
        ax[1].errorbar(xi, yi, yerr=ei, marker='o', color=ci,mfc='none',label='CTAX-Warming')
    elif idx ==len(temp)-1:
    # else:
        ax[1].errorbar(xi, yi, yerr=ei, marker='o', color=ci,mfc='none',label='CTAX-Cooling')
    else:
        ax[1].errorbar(xi, yi, yerr=ei, marker='o', color=ci,mfc='none')
ax[1].set_xlabel('Temperature (K)',fontsize=14)
ax[1].set_ylabel('Integrated intensity', fontsize=14)
ax[1].set_xlim(left=0,right=7)
ax[1].set_ylim(bottom=-0.5,top=8.5)
print("-------------------")
print(max(int)*1.1)
ax[1].minorticks_on()
# ax.errorbar(temp, int, yerr =unc_I,  label='intensities',marker='o', linestyle='None',color=labels)
for a in ax:
    a.tick_params(axis='both', labelsize=12)
ax[0].legend(loc='best',fontsize=5,ncol=2)
ax[1].annotate(
    '',  # no text
    xy=(2.1, 4),             # arrow tip (target) in data coords
    xytext=(3.3,4),     # arrow tail (start point)
    arrowprops=dict(arrowstyle='->', color='C0',linewidth=2)
)
ax[1].annotate(
    '',  # no text
    xy=(2.1, 3.5),             # arrow tip (target) in data coords
    xytext=(3.3, 3.5),     # arrow tail (start point)
    arrowprops=dict(arrowstyle='->', color='C1',linewidth=2)
)

plt.tight_layout()
# plt.show()

# ------ Add the OP from CNCS -----------
ax2 = ax[1].twinx()
temp_CNCS =np.array([0.25, 1.5,1.7, 3,3.5,3.7,4,5,6])
II_CNCS = np.array([0.3010245310337003, 0.29874539511031073, 0.3459544169537949, 0.25433368254234234, 0.2155043639102896, 0.14171625738029547, 0.06825918515215079, 0, 0])
err_CNCS = np.array([0.031812037760837755, 0.02923563427564466, 0.027750522462122264, 0.02145736774521768, 0.021916420627935185, 0.014881390490814753, 0.00851414483400446, 0, 0])
ax2.errorbar(temp_CNCS, II_CNCS*1.4, yerr=err_CNCS, fmt='s', label='CNCS (1 0 1)', color='C4')
ax2.tick_params(axis='y')
ax2.set_ylim(bottom=-0.033,top=0.5)
ax2.yaxis.set_major_locator(MultipleLocator(0.1))
ax2.minorticks_on()
ax2.annotate(
    '',  # no text
    xy=(5.5, 0.15),             # arrow tip (target) in data coords
    xytext=(4.3, 0.15),     # arrow tail (start point)
    arrowprops=dict(arrowstyle='->', color='C4',linewidth=2)
)

def piecewise_linear(t, ii, tn, beta, c):
    y = np.piecewise(t, [t < tn, t >= tn],
                     [lambda t:ii*(1-t/tn)**(2*beta)+c, lambda t:0*t+c])
    return y

# p0=[try_ii, try_tn, try_beta, c]
#     try_ii= 17.5
#     try_tn = 22
#     try_beta = 0.4
#     c = 0.05
p1 = [9.5,4.5,0.2,0.1]
# ax.plot(temp,piecewise_linear(temp,*p1))
temp_range= np.linspace(3,max(temp),1000)
# print(int)
parameters, covariance = curve_fit(piecewise_linear, temp, int,p0=p1)
std = np.sqrt(np.diag(covariance))
pp=[14,4.4,0.35,0.035]
# pp =[8.58073228,3.8876031,0.13366768,0.32909975]
# ax[1].plot(temp_range,piecewise_linear(temp_range,*parameters),linestyle='--')
# ax[1].axhline(y=piecewise_linear(2.5,12.5,4.35,0.3,0.035),xmin=0,xmax=1.5/6,linestyle='--',color='Red')
ax[1].plot(temp_range,piecewise_linear(temp_range,*pp),linestyle='--',color='Red',label='Guide line')
# print(parameters)
ax[1].legend(loc='best',fontsize=12)
ax[0].tick_params(which = 'both',direction='in')  # 'in', 'out', or 'inout'
ax[1].tick_params(which = 'both',direction='in')  # 'in', 'out', or 'inout'
ax2.tick_params(which = 'both',direction='in')  # 'in', 'out', or 'inout'


lines_1, labels_1 = ax[1].get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
plt.tight_layout()
# plt.show()
plt.savefig("Order_parameter.png", dpi=300, bbox_inches='tight')

#