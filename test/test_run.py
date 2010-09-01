from medusa import *

from pal import signal

x1,fs = signal.wavread("clean.wav")
x2 = signal.tone(400, fs, 10000)

y1 = np.ascontiguousarray(np.vstack((x1, x1*0)).swapaxes(0,1))
y2 = np.ascontiguousarray(np.vstack((x2*0, x2)).swapaxes(0,1))

pa.Pa_Initialize()

"""
a = ToneStream(1, 0, 440,  44100.0, 1.0)
b = ToneStream(1, 0, 1000, 44100.0, 1.0)
c = ToneStream(1, 0, 2100, 44100.0, 1.0)

d = ArrStream(y1, fs, 1.0)
e = ArrStream(y2, fs, 1.0)
"""

#printAvailableDevices("dsound")

"""
d = PaDevice(0, 2)
s1 = d.array_stream(y1, fs)
s2 = d.array_stream(y2, fs)

start_streams([s1,s2], True)
"""