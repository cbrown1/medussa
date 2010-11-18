import medussa
import numpy as np
from pal import signal
from time import sleep
import sys

fs = 44100
x,fs = medussa.readfile("clean.wav")
#x,fs = signal.wavread("speech-noise-tone.wav")

#y = np.ascontiguousarray(np.vstack((x, x)).swapaxes(0,1))

d = medussa.open_device()
#s = d.open_array(x,fs)
#s = d.open_file("clean.wav")
#s = d.create_tone(400,44100)
s = d.create_white(44100)

#s.arr = np.linspace(0, s.arr.size, s.arr.size).reshape(s.arr.shape)

#sy = d.open_array(y,fs)
#s = d.create_tone(440.0, 44100.0)


#s1 = d.create_tone(440.0, 44100.0)
#s2 = d.create_tone(880.0, 44100.0)
#fade = np.linspace(0.0, 1.0, 500)
#s1.play(), s2.play()
#for i in xrange(500):
#    s1.mix_mat = np.array([1.0-fade[i], fade[i]])
#    s2.mix_mat = np.array([fade[i], 1.0-fade[i]])
#    sleep(0.001)

#s = medussa.SndfileStream(d, "speech-noise-tone.wav")

def sweep_right(s, delta=0.001, steps=500):
    s.play()
    fade = np.linspace(0.0, 1.0, steps)
    for i in xrange(steps):
        s.mix_mat = np.array([1.0-fade[i], fade[i]])
        sleep(delta)
    s.pause()

def sweep_left(s, delta=0.001, steps=500):
    s.play()
    fade = np.linspace(0.0, 1.0, steps)
    for i in xrange(steps):
        s.mix_mat = np.array([fade[i], 1.0-fade[i]])
        sleep(delta)
    s.pause()
