#! /usr/bin/env python

import matplotlib.pyplot as plt
import readbruker as rb
import numpy as np
import scipy.fft as FT
from scipy.signal import find_peaks

File = '/home/cheva/Agua@Qn_Bruker/230516_Nacho_DQ/1'

# read in the bruker formatted data
dic, rawdata = rb.read(File)

TD = dic["acqus"]["TD"]
filter = 69
sw = dic["acqus"]["SW_h"]
dw = 1E6/sw # en microsegundos

data = rawdata[filter:]
data[0] = 2*data[0] # arreglo lo que el bruker rompe
nP = len(data)
t = np.array([dw*x for x in range(nP)])


plt.plot(t, data.real)
plt.plot(t, data.imag)
plt.show()



plt.plot(CS, spec.real, label='Spectrum (real)', color='coral')
# plt.fill_between(CS[mask], 0, spec.real[mask], label = fr'Peak area = {area_peak:.0f}', alpha = 0.25, color="teal")
# plt.plot(peaksx[0], peaksy[0] + 0.05, lw = 0, marker=11, color='black')
# plt.annotate(f'{peaksx[0]:.4f}', xy = (peaksx[0], peaksy[0] + 0.07), fontsize=30, ha='center') 
# plt.set_xlim(-0.2, 0.2)
# plt.set_ylim(-0.05, 1.2)
# plt.xaxis.set_minor_locator(AutoMinorLocator())
# plt.set_xlabel(r'$\delta$ [ppm]')
# plt.axvline(x=0, color='k', ls=':', lw=2)
# plt.axhline(y=0, color='k', ls=':', lw=2)
# plt.legend(loc='upper right')

plt.show()