# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Play a tone, fading from complete silence to full volume. The program will
continuously print out whether the fade is occuring or not, by calling
is_fading.
"""

import time
import numpy as np
import medussa as m


def main():
    d = m.open_default_device()
    tone = d.create_tone(500)
    tone.use_cosine_fades = True
    tone.mix_mat_fade_duration = 3.

    # Set up a dummy mix_mat array to work on
    silence_mix_mat = np.array([[0.0], [0.0]])
    full_volume_mix_mat = np.array([[1.0], [1.0]])
    tone.mix_mat = silence_mix_mat

    tone.play()

    tone.mix_mat = full_volume_mix_mat

    while True:
        print("Hit ctrl-c to end -- tone.is_fading = {val}".format(val=tone.is_fading))

if __name__ == " __main__":
    main()
