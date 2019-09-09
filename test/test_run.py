from __future__ import print_function

import medussa
import numpy as np
from time import sleep
import sys

try:
    # Python 2
    xrange
except NameError:
    # Python 3, xrange is now named range
    xrange = range


def main():
    fs = 44100.0
    x, fs = medussa.read_file("test/clean.wav")
    # x,fs = medussa.read_file("test/speech-noise-tone.wav")

    # y = medussa.write_wav("test/clean2.wav", x, fs, bits='u8')
    # y = medussa.write_ogg("test/clean2.ogg", x, fs)
    # y = medussa.write_flac("test/clean2.flac", x, fs, bits=16)
    # flac16 = formats.SF_FORMAT_FLAC[0] | formats.SF_FORMAT_PCM_16[0]
    # oggv = formats.SF_FORMAT_OGG[0] | formats.SF_FORMAT_VORBIS[0]
    # y = medussa.writefile("test/clean2.ogg", x, fs, format=oggv)
    # print y

    d = medussa.open_device()
    dd = medussa.open_device()

    sleep(1)
    sa = d.open_array(x, fs)
    sf = d.open_file("test/clean.wav")
    st = d.create_tone(400, fs)
    sw = d.create_white(fs)
    sp = d.create_pink(fs)
    sp2 = dd.create_pink(fs)

    # medussa.write_wav('test/clean2.wav', x, fs)

    def sweep_right(s, delta=0.001, steps=500):
        s.play()
        fade = np.linspace(0.0, 1.0, steps)
        for i in xrange(steps):
            s.mix_mat = np.array([1.0 - fade[i], fade[i]])
            sleep(delta)
        s.pause()

    def sweep_left(s, delta=0.001, steps=500):
        s.play()
        fade = np.linspace(0.0, 1.0, steps)
        for i in xrange(steps):
            s.mix_mat = np.array([fade[i], 1.0 - fade[i]])
            sleep(delta)
        s.pause()


if __name__ == "__main__":
    main()
