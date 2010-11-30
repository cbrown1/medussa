#include "medussa.h"


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

    printf("debug 1\n");
    gstate = PyGILState_Ensure();
    //
    // Start pulling values from calling object...
    //
    // `void *user_data`from `Stream.user_data`
    if (PyObject_HasAttrString(self, "user_data")) {
        attr = PyObject_GetAttrString(self, "user_data");
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
    printf("user_data: %p\n", user_data);
    printf("debug 2\n");

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
            Py_CLEAR(attr);
        }
    }
    else {
        return NULL;
    }

    printf("debug 3\n");

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

    // `unsigned long fpb` from `Stream.pa_fpb` [Frames per buffer]
    if (PyObject_HasAttrString(self, "pa_fpb")) {
        attr = PyObject_GetAttrString(self, "pa_fpb");
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

    printf("debug 4\n");

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
