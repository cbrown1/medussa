import medusa
import numpy as np
from pal import signal

x,fs = signal.wavread("clean.wav")

#y = np.ascontiguousarray(np.vstack((x, x*0)).swapaxes(0,1))

d = medusa.open_default_device()
s = d.open_array(x,fs)
