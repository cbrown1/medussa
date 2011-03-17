import medussa
import numpy as np
#from pal import signal
from time import sleep
import sys

fs = 44100.0
x,fs = medussa.readfile("test/speech-noise-tone.wav")
#x,fs = medussa.readfile("speech-noise-tone.wav")

#y = medussa.writewav("test/threetone.wav", x, fs/2)

#d = medussa.open_device(output_channels=2)
#d = medussa.open_device()

#sa = d.open_array(x,fs)
#sf = d.open_file("test/clean.wav")
#st = d.create_tone(400,fs)
#sw = d.create_white(fs)
#sp = d.create_pink(fs)


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
