import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')

data1 = np.genfromtxt('/Users/tianxionghan/research/TbV6Sn6/matlab_cal/TFIM_classical_3em6.txt', skip_header=1, unpack=True)
data2 = np.genfromtxt("/Users/tianxionghan/research/TbV6Sn6/matlab_cal/TFIM_classical_3em5.txt",skip_header=1, unpack=True)
h1 = data1[0]
mx1 = data1[-2]

h2 = data2[0]
mx2 = data2[-2]

fig, ax = plt.subplots()
ax.plot(h1,mx1,marker='.',label = 'B66 = 3e-6')
ax.plot(h2,mx2,marker='.',label = 'B66 = 3e-5')
print(max(h1))
ax.set_xlim(left=0,right=35)
ax.set_ylim(bottom=0,top=10)
ax.set_xlabel('Transverse Field (T)',fontsize=16)
ax.set_ylabel(r'$M_x\ (\mu_B)$',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(fontsize=14)
ax.axvline(x=26.8,color="C0",linestyle='--',ymin=0,ymax=0.9)
ax.axvline(x=26,color="C1",linestyle='--',ymin=0,ymax=0.9)
ax.text(23,8.3,"26T",fontsize=14)
ax.text(27,8.3,"27T",fontsize=14)
# ax.set_xticks([0,5,10,15,20,25,25.5,26.8,30,35])
plt.tight_layout()
plt.show()