import numpy as np
from matplotlib import pyplot as plt

path = '/Users/tianxionghan/research/CEF_cal/CrystalFieldCal/data/Field_measurement/'
file_list =['7','0','0p5','1','1p5',]
field = [7,0,0.5,1,1.5]
color_list=['lightgray','blue','red','C0','C1']
fig,ax= plt.subplots()
for idx, file_name in enumerate(file_list):
    file ='{}T_Q0p51p5_e0p02.xye'.format(file_name)
    data = np.genfromtxt(path + file, skip_header=1, unpack=True)
    e = data[0]
    I = data[1]
    dI = data[2]
    ax.errorbar(e,I,dI,fmt='.-',color = color_list[idx],label='{}T'.format(field[idx]))
ax.set_ylim(bottom=0,top = 0.002)
ax.set_xlim(left=0.5,right = 2)
ax.set_xlabel(r'$E (meV)$')
ax.set_ylabel(r'$I (r.u.)$')
plt.legend()
plt.tight_layout()
plt.show()
