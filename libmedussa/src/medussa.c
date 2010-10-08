#include "medussa.h"


PaStream *open_stream (PyObject *self)
{
    // Temporary local variables
    PaError err;
    PyObject *attr;

    // Variables for `Pa_OpenStream()`
    PaStream *stream;
    PaStreamParameters *spin;
    PaStreamParameters *spout;
    double fs;
    PaStreamCallback *callback_ptr;


    //
    // Start pulling values from calling object...
    //

    // `PaStream *stream` from `Stream.stream_ptr` 
    if (PyObject_HasAttrString(self, "stream_ptr")) {
        attr = PyObject_GetAttrString(self, "stream_ptr");
        if (attr == NULL) {
            return NULL;
        }
        Py_INCREF(attr);
        stream = (PaStream *) PyInt_AsLong(attr);
        Py_DECREF(attr);
    }
    else {
        return NULL;
    }

    // `PaStreamParameters *spin` from `Stream.in_param`
    if (PyObject_HasAttrString(self, "in_param")) {
        attr = PyObject_GetAttrString(self, "in_param");
        if (attr == NULL) {
            return NULL;
        }
        Py_INCREF(attr);
        spin = (PaStreamParameters *) PyInt_AsLong(attr);
        Py_DECREF(attr);
    }
    else {
        return NULL;
    }

    // `PaStreamParameters *spout` from `Stream.out_param`
    if (PyObject_HasAttrString(self, "out_param")) {
        attr = PyObject_GetAttrString(self, "out_param");
        if (attr == NULL) {
            return NULL;
        }
        Py_INCREF(attr);
        spout = (PaStreamParameters *) PyInt_AsLong(attr);
        Py_DECREF(attr);
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

    // `PaStreamCallback *callback_ptr` from `Stream.callback`
    if (PyObject_HasAttrString(self, "callback")) {
        attr = PyObject_GetAttrString(self, "callback");
        if (attr == NULL) {
            return NULL;
        }
        Py_INCREF(attr);
        callback_ptr = (PaStreamCallback *) PyInt_AsLong(attr);
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
                        paFramesPerBufferUnspecified,
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
