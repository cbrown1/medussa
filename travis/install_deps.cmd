:: Windows-only variant of before build script for cibuildwheel
python -m pip install --upgrade --only-binary=numpy numpy

:: TODO build portaudio/libsndfile instead of bundling in the repo?
