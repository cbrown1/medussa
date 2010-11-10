#include "medussa.h"


PaStream *open_stream (PyObject *self, PaStreamParameters *spin, PaStreamParameters *spout, PaStreamCallback *callback_ptr)
{
    // Temporary local variables
    PaError err;
    PyObject *attr;

    // Variables for `Pa_OpenStream()`
    PaStream *stream;
    //PaStreamParameters *spin;
    //PaStreamParameters *spout;
    double fs;
    unsigned long fpb;
    //PaStreamCallback *callback_ptr;

    //
    // Start pulling values from calling object...
    //

    // `PaStream *stream` from `Stream.stream_ptr` 
    if (PyObject_HasAttrString(self, "stream_ptr")) {
        attr = PyObject_GetAttrString(self, "stream_ptr");
        if (attr == NULL) {
            return NULL;
        }
        else if (attr == Py_None) {
            //printf("DEBUG: stream_ptr is none\n");
        }
        else {
            Py_INCREF(attr);
            err = PyInt_AsUnsignedLongMask(attr);
            stream = (PaStream *) PyInt_AsUnsignedLongMask(attr);
            Py_DECREF(attr);
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
        Py_DECREF(attr);
    }
    else {
        return NULL;
    }

    // `unsigned long fpb` from `Stream.pa_fpb` [Frames per buffer]
    if (PyObject_HasAttrString(self, "pa_fpb")) {
        attr = PyObject_GetAttrString(self, "pa_fpb");
        if (attr == NULL) {
            return NULL;
        }
        Py_INCREF(attr);
        fpb = PyInt_AsUnsignedLongMask(attr); // Only func in C API returning `unsigned long`
        Py_DECREF(attr);
    }
    else {
        return NULL;
    }
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
                        self);  
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

void test_msg ()
{
    printf("hello, world.\n");
}

void sndfile_as_ndarray (char *finpath, PyObject *arr)
{
    int frames_read;

    SNDFILE *fin;
    SF_INFO *finfo;
    //PyArrayObject *array;
    //array = (PyArrayObject *) arr;

    printf("finpath: %s\n", finpath);

    printf("debug 1\n");
    fin = sf_open(finpath, SFM_READ, finfo);
    printf("debug 2: %p\n", fin);
    frames_read = (int) sf_readf_double (fin, (double *) PyArray_DATA(arr), finfo->frames);
    printf("debug 3\n");
    sf_close(fin);
    printf("debug 4\n");
}