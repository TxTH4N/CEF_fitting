import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
from scipy.optimize import curve_fit
import scipy.special as spf
import scipy.constants as cts
from lmfit import Model, Parameters
from lmfit import create_params, fit_report, minimize
import matplotlib
import load_data as ld
import timeit
from pcf_lib.Operators import Ket
import renormalize_ratio as rr


x1 = np.linspace(-12, 12, 480)
x2 = np.linspace(-3, 3, 120)

y1 = +3.8775e-05 * x1**3 +0.0018964 * x1**2 -0.078275 * x1 +0.72305
y2 = +0.00030354 * x2**3 +0.0039009 * x2**2 -0.040862 * x2 +0.11303

fig, ax = plt.subplots()
ax.plot(x1, y1, '.',label = '$E_i$ = 12 meV Resolution')
ax.plot(x2, y2, '.',label = '$E_i$ = 3.32 meV Resolution')
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel('$\\Delta E$ (meV)',fontsize=14)
ax.set_ylabel('FHWM (meV)',fontsize=14)
ax.set_xlim(left=-12,right=12)
ax.set_ylim(bottom=0,top=2)
plt.legend(fontsize=14)
plt.show()