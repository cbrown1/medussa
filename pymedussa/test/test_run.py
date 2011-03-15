import medussa
import numpy as np
#from pal import signal
from time import sleep
import sys

from scipy.io import wavfile

fs = 44100.0
#x,fs = medussa.readfile("test/clean.wav")
#fs,x = wavfile.read("clean.wav")
#fs,x = wavfile.read("speech-noise-tone.wav")
#x = x * 1.0
#x = x / 32767.0
#y = np.hstack((x,x))
#x,fs = medussa.readfile("speech-noise-tone.wav")
#shape =x.shape
#x[:,0] = 1.0
#x[:,1] = 2.0
#x[:,2] = 3.0

#d = medussa.open_device(11)
d = medussa.open_device(output_channels=2)
#d = medussa.open_device()
#d.output_channels = 2
#sleep(1)
#s = d.open_array(x,fs)
s = d.open_file("test/clean.wav")
#s = d.create_tone(400,fs)
#s = d.create_white(fs)
#s.play()
#sleep(6)



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
