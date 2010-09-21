#include <Python.h>
#include <numpy/arrayobject.h>

#include <portaudio.h>

int callback_play_array (const void *,
                         void *,
                         unsigned long,
                         const PaStreamCallbackTimeInfo *,
                         PaStreamCallbackFlags,
                         void *);
int callback_play_tone  (const void *,
                         void *,
                         unsigned long,
                         const PaStreamCallbackTimeInfo *,
                         PaStreamCallbackFlags,
                         void *);
