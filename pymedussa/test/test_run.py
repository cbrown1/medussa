import medussa
import numpy as np
#from pal import signal
from time import sleep
import sys

from medussa.sndfile import formats

fs = 44100.0
x,fs = medussa.readfile("test/clean.wav")
#x,fs = medussa.readfile("test/speech-noise-tone.wav")

#y = medussa.writewav("test/clean2.wav", x, fs)
#flac16 = formats.SF_FORMAT_FLAC[0] | formats.SF_FORMAT_PCM_16[0]
#oggv = formats.SF_FORMAT_OGG[0] | formats.SF_FORMAT_VORBIS[0]
#y = medussa.writefile("test/clean2.ogg", x, fs, format=oggv)
#print y

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
