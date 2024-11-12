import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,10,1000)

def coth(x):
    return np.cosh(x)/np.sinh(x)

def brillouin(x,j):
    y = (2*j+1)/(2*j)*coth((2*j+1)/(2*j)*x)-1/(2*j)*coth(1/(2*j)*x)
    return y

field= np.linspace(1,10,1000)
e_pm =7.95
e_fm =7.78
shift_per_T = (8.251-8.071)/2

miub = 0.05788
kb=0.08617

j=6
g = 3/2
x = g*miub*j*field/(kb*8.5)

pm_f = e_pm+field*shift_per_T-brillouin(x,1/2)*(e_pm-e_fm)
fm_f = e_pm+field*shift_per_T-(e_pm-e_fm)
fig,ax = plt.subplots()
ax.plot(x,pm_f,label='PM')
ax.plot(x,fm_f,label='FM')
plt.legend()
plt.show()
# temp =
# print(e_pm-e0)
# field = np.linspace(0,8,800)
# temp =8

#
# y = j*g*field/(kb*temp)
#
# e_t = e_pm+g*miub*field+alpha*brillouin(y,1/2)
# plt.plot(field,e_t)
# plt.show()