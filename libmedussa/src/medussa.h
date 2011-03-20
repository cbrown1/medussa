#include <stdio.h>

#include <Python.h>
#include <numpy/arrayobject.h>

#include <portaudio.h>
#include <sndfile.h>

#include "medussa_callbacks.h"

#define ERROR_CHECK \
{ \
if (err < 0) { \
    PyErr_SetString(PyExc_RuntimeError, Pa_GetErrorText(err)); \
    return NULL; \
} \
}

#define NDARRAY_STREAM 0
#define SNDFILE_STREAM 1
#define TONE_STREAM 2

PaStream *open_stream (PyObject *,
                       PaStreamParameters *,
                       PaStreamParameters *,
                       PaStreamCallback *);

void start_streams (PaStream **, int);
int readfile_helper (SNDFILE *fin, double *arr, int frames);
int writefile_helper (char *foutpath, SF_INFO *finfo, double *arr, int format, int frames);
