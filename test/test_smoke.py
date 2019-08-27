from __future__ import print_function

import medussa
from os import path as _p

# This will at least verify Python/C API binding works and .so loading after install
x, fs = medussa.read_file(_p.join(_p.dirname(_p.abspath(__file__)), "clean.wav"))
assert fs == 44100
