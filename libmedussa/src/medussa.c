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
            printf("DEBUG: stream_p is none\n");
        }
        else {
            Py_INCREF(attr);
            printf("DEBUG 1-0-2-0: %d\n", PyInt_Check(attr));
            printf("DEBUG 1-0-2-1: %d\n", attr == Py_None);
            err = PyInt_AsUnsignedLongMask(attr);
            stream = (PaStream *) PyInt_AsUnsignedLongMask(attr);
            Py_DECREF(attr);
        }
    }
    else {
        return NULL;
    }

    /*
    // `PaStreamParameters *spin` from `Stream.in_param`
    if (PyObject_HasAttrString(self, "in_param")) {
        attr = PyObject_GetAttrString(self, "in_param");
        if (attr == NULL) {
            return NULL;
        }
        else if (attr == Py_None) {
            printf("DEBUG: in_param was None\n");
        }
        else {
            Py_INCREF(attr);
            spin = (PaStreamParameters *) PyInt_AsUnsignedLongMask(attr);
            Py_DECREF(attr);
        }
    }
    else {
        return NULL;
    }
    */

    /*
    // `PaStreamParameters *spout` from `Stream.out_param`
    if (PyObject_HasAttrString(self, "out_param")) {
        attr = PyObject_GetAttrString(self, "out_param");
        if (attr == NULL) {
            return NULL;
        }
        printf("DEBUG: out_param == None? %d\n", attr == Py_None);
        Py_INCREF(attr);
        spout = (PaStreamParameters *) attr;
        Py_DECREF(attr);
    }
    else {
        return NULL;
    }
    */

    printf("DEBUG: channel count: %d\n", spout->channelCount);

    // `double fs` from `Stream.fs`
    if (PyObject_HasAttrString(self, "fs")) {
        attr = PyObject_GetAttrString(self, "fs");
        if (attr == NULL) {
            return NULL;
        }
        printf("DEBUG: fs is PyFloat: %d\n", PyFloat_Check(attr));
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
