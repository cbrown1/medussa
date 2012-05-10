#include "medussa.h"

#if PY_MAJOR_VERSION >= 3
  #define PyInt_AsUnsignedLongMask PyLong_AsUnsignedLongMask
#endif

PaStream *open_stream (PyObject *self, PaStreamParameters *spin, PaStreamParameters *spout, PaStreamCallback *callback_ptr)
{
    // Temporary local variables
    PaError err;
    PyObject *attr;

    PyGILState_STATE gstate;

    // Variables for `Pa_OpenStream()`
    PaStream *stream;
    //PaStreamParameters *spin;
    //PaStreamParameters *spout;
    double fs;
    unsigned long fpb;
    void *user_data;
    //PaStreamCallback *callback_ptr;

    gstate = PyGILState_Ensure();
    //
    // Start pulling values from calling object...
    //
    // `void *user_data`from `Stream.user_data`
    if (PyObject_HasAttrString(self, "_callback_user_data")) {
        attr = PyObject_GetAttrString(self, "_callback_user_data");
        if (attr == NULL) {
            return NULL;
        }
        else if (attr == Py_None) {
            //printf("DEBUG: `user_data` is none\n");
        }
        else {
            Py_INCREF(attr);
            err = PyInt_AsUnsignedLongMask(attr);
            user_data = (void *) PyInt_AsUnsignedLongMask(attr);
            Py_CLEAR(attr);
        }
    }
    else {
        return NULL;
    }

    // `PaStream *stream` from `Stream.stream_ptr`
    if (PyObject_HasAttrString(self, "_stream_ptr")) {
        attr = PyObject_GetAttrString(self, "_stream_ptr");
        if (attr == NULL) {
            return NULL;
        }
        else if (attr == Py_None) {
            //printf("DEBUG: _stream_ptr is none\n");
        }
        else {
            Py_INCREF(attr);
            err = PyInt_AsUnsignedLongMask(attr);
            stream = (PaStream *) PyInt_AsUnsignedLongMask(attr);
            Py_CLEAR(attr);
        }
    }
    else {
        return NULL;
    }

    // `double fs` from `Stream.fs`
    if (PyObject_HasAttrString(self, "fs")) {
        attr = PyObject_GetAttrString(self, "fs");
        if (attr == NULL) {
            return NULL;
        }
        Py_INCREF(attr);
        fs = PyFloat_AsDouble(attr);
        Py_CLEAR(attr);
    }
    else {
        return NULL;
    }

    // `unsigned long fpb` from `Stream._pa_fpb` [Frames per buffer]
    if (PyObject_HasAttrString(self, "_pa_fpb")) {
        attr = PyObject_GetAttrString(self, "_pa_fpb");
        if (attr == NULL) {
            return NULL;
        }
        Py_INCREF(attr);
        fpb = PyInt_AsUnsignedLongMask(attr); // Only func in C API returning `unsigned long`
        Py_CLEAR(attr);
    }
    else {
        return NULL;
    }
    PyGILState_Release(gstate);
    //
    // ...end pulling values from calling object.
    //

    // Attempt to open the stream
    err = Pa_OpenStream(&stream,
                        spin,
                        spout,
                        fs,
                        fpb,
                        paNoFlag,
                        callback_ptr,
                        user_data);
    ERROR_CHECK;

    // Return the new integer value of the mutated `PaStream *` back to Python
    return stream;
}

void start_streams (PaStream *stream_array[], int num_streams)
{
    int i;

    for (i = 0; i < num_streams; i++) {
        Pa_StartStream(stream_array[i]);
    }
}

int readfile_helper (SNDFILE *fin, double *arr, int frames)
{
    int frames_read;
    int err;

    err = sf_seek(fin, 0, 0);
    if (err != 0) { return err; }

    frames_read = sf_readf_double(fin, arr, frames);
    return frames_read;
}

int writefile_helper (char *foutpath, SF_INFO *finfo, double *arr, int format, int frames)
{
    int frames_written = 0;
    SNDFILE *fout;

    finfo->format = format;

    if (!sf_format_check(finfo)) {
        printf("Bad SF_INFO struct.\n");
        return -1;
    }

    fout = sf_open(foutpath, SFM_WRITE, finfo);

    frames_written = sf_writef_double (fout, arr, frames);

    sf_close(fout);

    return frames_written;
}
