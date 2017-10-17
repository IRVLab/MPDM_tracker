import sys
import numpy as np
import cv2
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# Number of samplepoints
N = 15
f_s = 15
# sample spacing
T = 1.0
f0 = (f_s)/N
print(f0)
x = np.linspace(0.0, N*T, N)
y = np.sin(2.0 * 2.0*np.pi*x)
yf = scipy.fftpack.fft(y)
xf = np.linspace(0, N/2, (N/2)*f0)

print(xf, N/2)
fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()
    






