#include "medusa.h"

#include <stdio.h>
#include <math.h>

#define TWOPI 6.2831853071795862

int callback_play_array (const void *pa_buf_in, void *pa_buf_out,
                         unsigned long frames,
                         const PaStreamCallbackTimeInfo *time_info,
                         PaStreamCallbackFlags status_flags,
                         void *user_data)
{

    float *output;
    PyArrayObject *x;

    static int samp = 0;
    static int chan = 0;

    int i;
    int samples;

    output = (float *) pa_buf_out;

    x = (PyArrayObject *) user_data;

    frames = frames;
    samples = (int) (frames * PyArray_DIM(x, 1));

    for (i = 0; i < samples; i++) {
        output[i] = 0.0;
    }
    i = 0;

    for (samp; samp < PyArray_DIM(x, 0); samp++) {
        for (chan; chan < PyArray_DIM(x, 1); chan++) {
            if (i < samples) {
                output[i] = (float) *((double *) PyArray_GETPTR2(x, samp, chan));
                i++;
            }
            else {
                return paContinue;
            }
        }
        chan = 0;
    }

    return paComplete;
}


int callback_play_tone  (const void *pa_buf_in, void *pa_buf_out,
                         unsigned long frames,
                         const PaStreamCallbackTimeInfo *time_info,
                         PaStreamCallbackFlags status_flags,
                         void *user_data)
{
    static int t;

    int i;
    float *output;
    float freq;

    if (user_data == NULL) {
        freq = 440.0;
    }
    else {
        freq = *((float *) user_data);
    }

    t = 0;

    output = (float *) pa_buf_out;

    for (i = 0; i < frames; i++, t++) {
        output[i] = sin(TWOPI * freq * t );
    }

    return paContinue;
}


