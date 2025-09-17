import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import PyCrystalField as cef
matplotlib.use('QtAgg')

TVSLig, Tb = cef.importCIF('/Users/tianxionghan/research/CEF_cal/CrystalFieldCal/refineCIF/TbV6Sn6_edit.cif')

Tb.printEigenvectors()

HVSLig, Ho = cef.importCIF('/Users/tianxionghan/research/CEF_cal/CrystalFieldCal/refineCIF/HoV6Sn6_edit.cif')

Ho.printEigenvectors()

print(Tb.B)
print(Ho.B)