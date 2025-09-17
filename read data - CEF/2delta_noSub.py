import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')

temp = np.array([1.47, 2.11, 2.62, 3.11, 3.6,  4.1,  4.59, 5.08, 5.58, 6.07, 6.58,
                6.08, 5.58, 5.09, 4.59, 4.35, 4.1,  3.86, 3.61, 3.37, 3.12,
                2.88, 2.64, 2.4,  1.49])
op = np.array([7.11, 7.49, 6.83, 5.91, 4.28, 1.59, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.44, 1.95, 2.70, 3.86, 5.78, 6.26, 6.56, 6.43, 7.42, 7.93])
op_err = np.array([0.20, 0.25, 0.19, 0.25, 0.21, 0.22, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.23, 0.24, 0.22, 0.24, 0.20, 0.25, 0.22, 0.21, 0.24, 0.22])

idx_0 = np.argsort(temp)
temp_sort_0 = temp[idx_0]
op_sort = op[idx_0]
op_err_sort=op_err[idx_0]
log_op = np.log(op_sort)
# fig,ax = plt.subplots()
# ax.errorbar(temp, op,yerr=op_err,fmt='o',ecolor='black')

add_temp = np.array([])
delta_cen = np.array([])
delta_err = np.array([])
temperature_list = np.array([0.25, 1.5, 3.5, 3.7])
cen_1 = np.array([1.29392, 1.29486, 1.30562, 1.32541])
cen_2 = np.array([1.63807, 1.63859, 1.62332, 1.60550])
cen_1_err = np.array([0.00132, 0.00127, 0.00172, 0.00335])
cen_2_err = np.array([0.00106, 0.00114, 0.00190, 0.00334])
add_temp=np.append(add_temp,temperature_list)
delta_cen=np.append(delta_cen,cen_2-cen_1)
delta_err=np.append(delta_err,np.sqrt(cen_1_err**2+cen_2_err**2))


temperature_list_old = np.array([1.7, 3, 4, 5, 6, 8, 10])
cen_1_old=np.array([1.30051, 1.30937, 1.35560, 1.38026, 1.38872, 1.38396, 1.38810])
cen_2_old = np.array([1.64273, 1.63012, 1.58100, 1.54837, 1.55377, 1.54968, 1.54464])
cen_1_old_err = np.array([0.00180, 0.00206, 0.00469, 0.00777, 0.00369, 0.00752, 0.00489])
cen_2_old_err = np.array([0.00158, 0.00214, 0.00410, 0.00940, 0.00367, 0.00689, 0.00455])
add_temp=np.append(add_temp,temperature_list_old)
delta_cen=np.append(delta_cen,cen_2_old-cen_1_old)
delta_err=np.append(delta_err,np.sqrt(cen_1_old_err**2+cen_2_old_err**2))


temperature_list_ht = np.array([10, 12.5, 15, 17.5])
cen_1_ht = np.array([1.38561, 1.39078, 1.39402, 1.37934])
cen_2_ht = np.array([1.53716, 1.53577, 1.54661, 1.54687])
cen_1_ht_err = np.array([0.00446, 0.00440, 0.00457, 0.00857])
cen_2_ht_err = np.array([0.00530, 0.00572, 0.00570, 0.00877])
add_temp=np.append(add_temp,temperature_list_ht)
delta_cen=np.append(delta_cen,cen_2_ht-cen_1_ht)
delta_err=np.append(delta_err,np.sqrt(cen_1_ht_err**2+cen_2_ht_err**2))

idx = np.argsort(add_temp)
temp_sort = add_temp[idx]
delta_sort = (delta_cen[idx])  ##original
err_sort = (delta_err[idx])
# delta_sort = (delta_cen[idx]-0.15)  ##/0.024

log_delta=np.log(delta_sort)
# print(len(log_op))
# print(len(log_delta))
# mask = temp_sort >= 5
# average = delta_sort[mask].mean()
# print(average)  #0.156

fig,ax = plt.subplots()
ax.errorbar(temp_sort,delta_sort,yerr =err_sort ,marker='o',label='2$\delta$')
ax.errorbar(temp_sort_0,op_sort*0.024,yerr =op_err_sort*0.024 ,marker='o',label='0.024*Order parameter')
# ax.plot(temp_sort,log_delta,marker='o',label='2$\delta$')
# ax.plot(temp_sort_0,2*log_op+np.log(0.004),marker='o',label='Order parameter')
# ax.plot(temp_sort_0,log_op+np.log(0.024),marker='o',label='Order parameter')
# ax.plot(temp_sort_0,0.5*log_op+np.log(0.07),marker='o',label='0.5*Order parameter')

# print(log_delta)
# print(log_op)
# ax.set_xlabel('Temperature (K)')
# plt.legend()
# plt.show()


### 2delta vs I below
# 1. define two sets
from scipy.interpolate import interp1d
print('OP temperature: {}'.format(temp_sort_0))  #
print('OP: {}'.format(op_sort))  # Y1
print('2d temperature: {}'.format(temp_sort))
print('2d temperature: {}'.format(delta_sort))   # Y2
temp1=temp_sort_0
Y1 = op_sort
temp2= temp_sort
Y2 = delta_sort
Y2_err = err_sort

# 2. mask and interpolate
mask = (temp1 >= temp2.min()) & (temp1 <= temp2.max())
t_common = temp1[mask]
Y1_common = Y1[mask]
Y1_common_err = op_err_sort[mask]
print('common temperature: {}'.format(t_common))
print('magnetization in range: {}'.format(Y1_common))
interp_func = interp1d(temp2, Y2, kind='linear', bounds_error=False, fill_value=np.nan)
Y2_interp = interp_func(t_common)
interp_err = interp1d(temp2, Y2_err, kind='linear', bounds_error=False, fill_value=np.nan)
Y2_err_interp = interp_err(t_common)
print('2 delta after interpolate: {}'.format(Y2_interp))
ax.errorbar(t_common,Y2_interp,yerr = Y2_err_interp,marker='o',linestyle = 'None',label='interpoled 2delta')
ax.set_xlabel('Temperature (K)')
# ax.set_ylabel('Temperature (K)')
plt.legend()
# plt.show()

# 3. Plot Y1 vs Y2
fig2,ax2 = plt.subplots()
ax2.errorbar( Y1_common,Y2_interp,xerr=Y1_common_err,yerr = Y2_err_interp, marker='o',linestyle = 'None')
ax2.set_xlabel("order parameter")
ax2.set_ylabel("2 delta")
# plt.title("Y1 vs Y2 (aligned via temp1)")
plt.grid(True)
# plt.show()

# delta_0=np.average(Y2_interp[-7:])
# delta_0_err = np.sqrt(np.sum(Y2_err_interp[-7:]**2))
# print('delta_0: {}'.format(delta_0))
delta_0=0
delta_0_err=0
cut_idx=-2
ax2.errorbar( Y1_common[:cut_idx+1],Y2_interp[:cut_idx+1],yerr=0, marker='o',linestyle='None')
ax2.errorbar( Y1_common[:cut_idx],Y2_interp[:cut_idx],yerr=0, marker='o',linestyle='None')
log_2d = np.log(Y2_interp[:cut_idx]-delta_0)
log_2d_err = np.sqrt(Y2_err_interp[:cut_idx]**2+delta_0_err**2)/Y2_interp[:cut_idx]
log_mag = np.log(Y1_common[:cut_idx])
log_mag_err =Y1_common_err[:cut_idx]/Y1_common[:cut_idx]


fig3,ax3 = plt.subplots()
print(log_mag)
print(log_2d)
ax3.errorbar( log_mag,log_2d,xerr =log_mag_err,yerr=log_2d_err,marker= 'o',linestyle='None')
# ax3.loglog( log_mag,log_2d,marker= 'o',linestyle='None')
ax3.set_ylabel("Ln( 2 delta - {:.4f})".format(delta_0))
ax3.set_xlabel("Ln( II)")
# plt.title("Y1 vs Y2 (aligned via temp1)")
# plt.grid(True)
# plt.show()

# fit the log-log plots

from lmfit import Model

def line(x, slope,intercept):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return slope*x+intercept

fmod = Model(line)
params = fmod.make_params()
params['slope'].set(1,vary=True)
params['intercept'].set(-3)
from numpy import gradient
dy_dx = gradient(log_2d, log_mag)
sigma_y_eff = np.sqrt(log_2d_err**2 + (dy_dx * log_mag_err)**2)
print(sigma_y_eff)
result = fmod.fit(log_2d, params, x=log_mag, sigma=sigma_y_eff)
print(result.fit_report())
# fitted_params=result.params.items()
fitted_params = [param.value for name, param in result.params.items()]
print(fitted_params)
print('-------------------------------')
print('Parameter    Value       Stderr')
for name, param in result.params.items():
    print(f'{name:7s} {param.value:11.5f} {param.stderr:11.5f}')
ax3.plot(log_mag, result.best_fit, linestyle='-', label='best fit: {:.2f}X-{:.2f}'.format(*fitted_params))
# ax3.plot(log_mag,0.5*log_mag-2.83729,linestyle='--', label='0.5X-2.84')
ax3.grid(alpha=0.6)
# Setting the scale of the x-axis to logarithmic
# ax3.set_xscale('log', base=np.e)
# ax3.text(-0.5,-2, 'y = {}x{}'.format())
# x_ticks = np.linspace(log_mag.min(),log_mag.max(),5)
# y_ticks = np.linspace(log_2d.min(),log_2d.max(),5)
# ax3.set_xticks(x_ticks)
# ax3.set_xticklabels([f"$\\ln({np.exp(t):.2f})$" for t in x_ticks])
#
# ax3.set_yticks(y_ticks)
# ax3.set_yticklabels([f"$\\ln({np.exp(t):.2f})$" for t in y_ticks])

plt.legend()
plt.show()
