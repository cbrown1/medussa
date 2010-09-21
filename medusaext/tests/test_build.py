import os
import numpy as np
from pal import signal
import _medusaext

from multiprocessing import Process

def play(x,fs):
    p = Process(target = _medusaext.play_array, args=(x,fs))
    p.start()


if __name__ == "__main__":
    ch1,fs = signal.wavread("clean.wav")
    ch1 = ch1
    ch2 = signal.tone(1000,fs,10000)
    ch3 = signal.tone(400,fs,10000)
    #ch1 = np.ones(8)
    #ch2 = np.ones(8) * 2
    #ch3 = np.ones(8) * 3

    #x = np.vstack((ch1,ch2,ch3)).swapaxes(0,1)
    x = np.vstack((ch3*0, ch3)).swapaxes(0,1)
    y = np.vstack((ch1*0,ch1)).swapaxes(0,1)
    z = np.vstack((ch2,ch2*0)).swapaxes(0,1)


    #m = _medusaext


    #m.wrap_Pa_Initialize()
    #m.play_array(x,fs)
    #m.play_array(y,fs)

    #m.wrap_Pa_Initialize()
    #m.play_array(x,fs)