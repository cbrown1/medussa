import medussa
from os import path as _p

TEST_DIR = _p.dirname(_p.abspath(__file__))


def test_smoke_read_file():
    """This will at least verify Python/C API binding works and .so loading after install."""
    x, fs = medussa.read_file(_p.join(TEST_DIR, "clean.wav"))
    assert fs == 44100
