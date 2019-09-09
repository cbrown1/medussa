# -*- coding: utf-8 -*-
from __future__ import print_function

"""
On multichannel hardware, this will play a tone and update mix_mat to simulate
motion. It will cycle through all the channels and loop back around again indefinitely.
On stereo (2-channel) hardware, this results in left-right-left...
"""

import time
import numpy as np
import medussa as m


def main():
    d = m.open_default_device()
    tone = d.create_tone(500)
    tone.mix_mat_fade_duration = 1.

    # "Speaker array". For multichannel hardware, hardcode this with the number of
    # actual channels on the device (medussa always sets this to 2 because
    # portaudio reports this value unreliably.
    sarray = range(d.out_channels)
    # Select first channel
    s = sarray[0]
    looping = True
    # Set up a dummy mix_mat array to work on
    mmlocal = np.array([[.25], [0.]])
    tone.mix_mat = mmlocal
    tone.play()

    print("Hit ctrl-c to end")

    # This loop turns on the next channel, turns off the current channel,
    # (wrapping when needed), and waits until fading is done.
    while looping:
        print(s+1)
        # Get next channel
        s1 = s+1
        if s1 > sarray[-1]:
            s1 = sarray[0]
        # Turn next channel on
        mmlocal[s1][0] = .25
        # Turn last channel off
        mmlocal[s][0] = 0.
        # Update mix_mat
        tone.mix_mat = mmlocal
        # Wait for fade to finish
        time.sleep(tone.mix_mat_fade_duration)
        # Get ready for next channel
        s = s1


if __name__ == "__main__":
    main()
