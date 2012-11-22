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

#include <stdio.h>

#include <Python.h>
#include <numpy/arrayobject.h>

#include <portaudio.h>
#include <sndfile.h>

#include "medussa_callbacks.h"

//Requires that the function using this macro has a local PyGILState_STATE
//variable called gstate (e.g., for use with PyGILState_Ensure and
//PyGILState_Release).

#define ERROR_CHECK \
{ \
if (err < 0) { \
    PyErr_SetString(PyExc_RuntimeError, Pa_GetErrorText(err)); \
    PyGILState_Release(gstate); \
    return NULL; \
} \
}

PaStream *open_stream (PyObject *,
                       PaStreamParameters *,
                       PaStreamParameters *,
                       PaStreamCallback *);

void start_streams (PaStream **, int);
int readfile_helper (SNDFILE *fin, double *arr, int frames);
int writefile_helper (char *foutpath, SF_INFO *finfo, double *arr, int format, int frames);
