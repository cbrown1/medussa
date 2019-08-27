/*
# Copyright (c) 2010-2012 Christopher Brown
#
# This file is part of Medussa.
#
# Medussa is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Medussa is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Medussa.  If not, see <http://www.gnu.org/licenses/>.
#
# Comments and/or additions are welcome. Send e-mail to: cbrown1@pitt.edu.
#
*/

#include "medussa.h"
#include "log.h"

#if PY_MAJOR_VERSION >= 3
  #define PyInt_AsVoidPtr(p) PyLong_AsVoidPtr(p)
  #define PyInt_AsUnsignedLongMask PyLong_AsUnsignedLongMask
#else
  #define PyInt_AsVoidPtr(p) ((void *)PyInt_AsUnsignedLongLongMask(p))
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
    void *user_data = NULL;
    //PaStreamCallback *callback_ptr;

    gstate = PyGILState_Ensure();
    //
    // Start pulling values from calling object...
    //
    // `void *user_data`from `Stream.user_data`
    if (PyObject_HasAttrString(self, "_callback_user_data")) {
        attr = PyObject_GetAttrString(self, "_callback_user_data");
        if (attr == NULL) {
            error("no _callback_user_data");
            PyGILState_Release(gstate);
            return NULL;
        }
        else if (attr == Py_None) {
            debug("_callback_user_data == Py_None");
        }
        else {
            Py_INCREF(attr);
            user_data = PyInt_AsVoidPtr(attr);
            assert(!PyErr_Occurred());
            /* XXX PyInt_ APIs signal errors with -1 value which never is a valid pointer */
            assert(user_data != (void*)-1 && user_data != NULL);
            if (PyErr_Occurred()) {
                error("failed reading _callback_user_data from self");
                PyErr_Print();
                return NULL;
            }
            Py_CLEAR(attr);
        }
    }
    else {
        PyGILState_Release(gstate);
        return NULL;
    }

    // `PaStream *stream` from `Stream.stream_ptr`
    if (PyObject_HasAttrString(self, "_stream_ptr_addr")) {
        attr = PyObject_GetAttrString(self, "_stream_ptr_addr");
        if (attr == NULL) {
            error("no _stream_ptr_addr");
            PyGILState_Release(gstate);
            return NULL;
        }
        else if (attr == Py_None) {
            // debug("_stream_ptr_addr == Py_None");
        }
        else {
            Py_INCREF(attr);
            stream = (PaStream *) PyInt_AsVoidPtr(attr);
            assert(!PyErr_Occurred());
            assert(stream != (void*)-1);
            Py_CLEAR(attr);
        }
    }
    else {
        PyGILState_Release(gstate);
        return NULL;
    }

    // `double fs` from `Stream.fs`
    if (PyObject_HasAttrString(self, "fs")) {
        attr = PyObject_GetAttrString(self, "fs");
        if (attr == NULL) {
            PyGILState_Release(gstate);
            return NULL;
        }
        Py_INCREF(attr);
        fs = PyFloat_AsDouble(attr);
        Py_CLEAR(attr);
    }
    else {
        PyGILState_Release(gstate);
        return NULL;
    }

    // `unsigned long fpb` from `Stream._pa_fpb` [Frames per buffer]
    if (PyObject_HasAttrString(self, "_pa_fpb")) {
        attr = PyObject_GetAttrString(self, "_pa_fpb");
        if (attr == NULL) {
            PyGILState_Release(gstate);
            return NULL;
        }
        Py_INCREF(attr);
        fpb = PyInt_AsUnsignedLongMask(attr); // Only func in C API returning `unsigned long`
        assert(!PyErr_Occurred());
        Py_CLEAR(attr);
    }
    else {
        PyGILState_Release(gstate);
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
                        user_data);

    ERROR_CHECK;

    PyGILState_Release(gstate);

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
    debug("BEFORE sf_seek");
    err = sf_seek(fin, 0, 0);
    if (err != 0) {
        error("sf_seek=%d", err);
        return err;
    }

    debug("BEFORE sf_readf_double");
    frames_read = sf_readf_double(fin, arr, frames);
    debug("sf_readf_double=%d", frames_read);
    return frames_read;
}

int writefile_helper (char *foutpath, SF_INFO *finfo, double *arr, int format, int frames)
{
    int frames_written = 0;
    SNDFILE *fout;

    finfo->format = format;

    if (!sf_format_check(finfo)) {
        error("Bad SF_INFO struct");
        return -1;
    }

    fout = sf_open(foutpath, SFM_WRITE, finfo);

    frames_written = sf_writef_double (fout, arr, frames);

    sf_close(fout);

    return frames_written;
}

#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, 0 }; \
          ob = PyModule_Create(&moduledef)
#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) void init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          ob = Py_InitModule3(name, methods, doc)
#endif

static PyMethodDef module_functions[] = {
    {NULL, NULL}
};

extern MOD_INIT(libmedussa);
MOD_INIT(libmedussa) {
    PyObject *m;
    debug("ENTER initlibmedussa");
    MOD_DEF(m, "cmedussa", "medussa C extension library", module_functions);
    if (m == NULL) {
        error("PyModuleCreate/PyInitModule3 failed");
        return MOD_ERROR_VAL;
    }
    debug("RETURN initlibmedussa=%p", m);
    return MOD_SUCCESS_VAL(m);
}
