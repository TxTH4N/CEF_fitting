import numpy as np
import matplotlib.pyplot as plt

file = 'data.txt'
with open(file, 'r') as f:
    data = np.genfromtxt(file, skip_header=1, unpack=True)
    tem = data[0]
    ent = data[1]*8.314
    sph = data[2]
print(ent)
# fig1,ax1= plt.subplots()
# ax1.plot(tem, ent, 'o', markersize=7)
# ax1.set_xlabel('Temperature (K)',fontsize=20)
# ax1.set_ylabel('Entropy',fontsize=20)
# ax1.tick_params(axis='both', which='major', labelsize=20)
# ax1.axhline(y=8.314*np.log(2), color='r', linestyle='--')
# ax1.set_xlim([0,10])
# ax1.set_ylim([-1,7.5])
#
# left, bottom, width, height = [0.58, 0.3, 0.35, 0.35]
# ax2 = fig1.add_axes([left, bottom, width, height])
# ax2.plot(tem, ent, 'o', markersize=3)
# ax2.set_xlabel('Temperature (K)',fontsize=13)
# ax2.set_ylabel('Entropy',fontsize=13)
# ax2.tick_params(axis='both', which='major', labelsize=13)
# ax2.axhline(y=8.314*np.log(13), color='b', linestyle='--')

def anomaly (x,dlt):
    a = 1
    y = 8.314*(dlt/x)**2*(a*np.exp(dlt/x))/(1+np.exp(dlt/x))**2
    return y
fig3,ax3= plt.subplots()
ax3.plot(tem, sph, 'o', markersize=5)
ax3.set_xlabel('Temperature (K)',fontsize=20)
ax3.set_ylabel('$C_p/T$ $(J\cdot mol^{-1}\cdot K^2)$',fontsize=20)
ax3.tick_params(axis='both', which='major', labelsize=20)
# ax3.plot(tem, anomaly(tem,10),'-')
# ax3.axhline(y=8.314*np.log(2), color='r', linestyle='--')
# ax3.set_xlim([0,10])
# ax3.set_ylim([-1,7.5])

plt.tight_layout()
plt.show()