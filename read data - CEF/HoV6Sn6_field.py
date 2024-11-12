import numpy as np
import matplotlib.pyplot as plt
a0 = 0
a1 = 0
b = 0.03301

muB = 0.05788
B_mf = np.linspace(0.0, 1.0, 100)
field_a0 = 5/4*muB*B_mf*(7*0.791**2+1*0.57**2-5*0.221**2)
field_a1 = 5/4*muB*B_mf*(-7*0.791**2-1*0.57**2+5*0.221**2)
field_b = 5/4*muB*B_mf*(0)

a0h = a0+field_a0
a1h = a1+field_a1
bh = b+field_b
fig,ax = plt.subplots()
ax.plot(B_mf,a0h)
ax.plot(B_mf,a1h)
ax.plot(B_mf,bh)
ax.set_xlabel(r'$B_mf$ (T)',fontsize=15)
ax.set_ylabel('Energy (meV)',fontsize=15)
ax.set_xlim(left=0,right=1)
plt.show()