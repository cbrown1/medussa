#include "medussa.h"

#include <stdio.h>
#include <math.h>

#define TWOPI 6.2831853071795862

#define ERROR_CHECK \
{ \
if (err < 0) { \
    PyErr_SetString(PyExc_RuntimeError, Pa_GetErrorText(err)); \
    return NULL; \
} \
}


int callback_ndarray (const void *pa_buf_in, void *pa_buf_out,
                      unsigned long frames,
                      const PaStreamCallbackTimeInfo *time_info,
                      PaStreamCallbackFlags status_flags,
                      void *user_data)
{
    float *buf_out;           // Points to `pa_buf_out`
    ContigArrayHandle *carrh; // Points to `user_data`

    PyArrayObject *x;
    int chan_i;
    int samp_i;

    int i;
    int buf_out_samples;

    int chan_size;
    int samp_size;

    buf_out = (float *) pa_buf_out;
    carrh = (ContigArrayHandle *) user_data;

    x = (PyArrayObject *) carrh->x;
    chan_i = carrh->chan_i;
    samp_i = carrh->samp_i;
    
    chan_size = PyArray_DIM(x, 1);
    samp_size = PyArray_DIM(x, 0);

    buf_out_samples = chan_size * ((int) frames);

    for (i = 0; i < buf_out_samples; i++) {
        buf_out[i] = 0.0;
    }
    i = 0;

    for (samp_i; samp_i < samp_size; samp_i++) {
        for (chan_i; chan_i < chan_size; chan_i++) {
            if (i < buf_out_samples) {
                buf_out[i] = (float) *((double *) PyArray_GETPTR2(x, samp_i, chan_i));
                i++;
            }
            else {
                carrh->chan_i = chan_i;
                carrh->samp_i = samp_i;
                return paContinue;
            }
        }
        chan_i = 0;
    }

    // Reset cursors for future playback
    carrh->chan_i = 0;
    carrh->samp_i = 0;

    if (carrh->loop) {
        return paContinue;
    }
    else {
        return paComplete;
    }
}


int callback_sndfile_read (const void *pa_buf_in, void *pa_buf_out,
                          unsigned long frames,
                          const PaStreamCallbackTimeInfo *time_info,
                          PaStreamCallbackFlags status_flags,
                          void *user_data)
{
    float *buf_out;   // Points to `pa_buf_out`
    SndfileData *sfd; // Points to `user_data`

    SNDFILE *fin;
    SF_INFO *fin_info;
    int frames_read;
    float scale;

    int i;
    int alpha;
    int beta;
    float *tmp_arr;

    buf_out = (float *) pa_buf_out;

    sfd = (SndfileData *) user_data;
    fin = (SNDFILE *) sfd->fin;
    fin_info = (SF_INFO *) sfd->fin_info;

    tmp_arr = (float *) malloc(sizeof(float) * frames * fin_info->channels);

    frames_read = sf_readf_float (fin, tmp_arr, frames);

    // Scale the output data buffer now that it's been copied
    scale = sfd->scale; // Using local var to avoid derefencing pointer in loop
    for (i = 0; i < (frames_read * fin_info->channels); i++) {
        tmp_arr[i] = tmp_arr[i] * scale;
    }

    // Now copy portions of each frame into the true output buffer
    alpha = fin_info->channels;
    beta = sfd->channel_count;
    for (i = 0; i < frames_read; i++) {
        memcpy(buf_out + i*beta, tmp_arr + i*alpha, sizeof(float)*alpha);
    }

    // Increment time counter by playback delta
    sfd->time += frames;

    if (frames_read == frames) {
        // Frames returned equals frames requested, so we didn't reach EOF
        return paContinue;
    }
    else {
        // We've reached EOF
        sf_seek(fin, 0, SEEK_SET); // Reset cursor to start of sound file
        if (sfd->loop) {
            return paContinue;
        }
        else {
            // We're really all done
            return paComplete;
        }
    }
}


int callback_tone  (const void *pa_buf_in, void *pa_buf_out,
                    unsigned long frames,
                    const PaStreamCallbackTimeInfo *time_info,
                    PaStreamCallbackFlags status_flags,
                    void *user_data)
{
    unsigned int t;
    ToneData *td;
    unsigned int i, j; // frame_iter, chan_iter, respectively
    float *buf_out;

    unsigned int channels;
    unsigned int chan_out;
    
    float tone_freq;
    float samp_freq;
    float scale;

    buf_out = (float *) pa_buf_out;

    td = (ToneData *) user_data;
    if (td == NULL) {
        return paComplete;
    }

    t = td->t;
    channels = td->channels;
    chan_out = td->chan_out;
    tone_freq = td->tone_freq;
    samp_freq = td->samp_freq;
    scale = td->scale;

    //printf("in tone callback at t = %d\n", t);

    for (i = 0; i < frames; i++) {
        for (j = 0; j < channels; j++) {
            buf_out[i*channels + j] = 0.0;
        }
        buf_out[i*channels + chan_out] = scale * sin(TWOPI * t / samp_freq * tone_freq);
        t++;
    }
    td->t = t;

    return paContinue;
}


int callback_pink  (const void *pa_buf_in, void *pa_buf_out,
                    unsigned long frames,
                    const PaStreamCallbackTimeInfo *time_info,
                    PaStreamCallbackFlags status_flags,
                    void *user_data)
{
    unsigned int t;
    ToneData *td;
    unsigned int i, j; // frame_iter, chan_iter, respectively
    float *buf_out;

    unsigned int channels;
    unsigned int chan_out;
    
    float tone_freq;
    float samp_freq;
    float scale;

    buf_out = (float *) pa_buf_out;

    td = (ToneData *) user_data;
    if (td == NULL) {
        return paComplete;
    }

    t = td->t;
    channels = td->channels;
    chan_out = td->chan_out;
    tone_freq = td->tone_freq;
    samp_freq = td->samp_freq;
    scale = td->scale;

    for (i = 0; i < frames; i++) {
        for (j = 0; j < channels; j++) {
            buf_out[i*channels + j] = 0.0;
        }
        buf_out[i*channels + chan_out] = scale * sin(TWOPI * t / samp_freq * tone_freq);
        t++;
    }
    td->t = t;

    return paContinue;
}


void start_streams (PaStream *stream_array[], int num_streams)
{
    int i;

    for (i = 0; i < num_streams; i++) {
        Pa_StartStream(stream_array[i]);
    }
}


PaStream *open_stream (PaStream *stream,
                       PaStreamParameters *in_param,
                       PaStreamParameters *out_param,
                       PyObject *self,
                       void *user_data,
                       int callback)
{
    PaError err;
    PaStreamCallback *callback_func;

    PyObject *attr;
    double samp_freq;

    switch (callback) {
    case NDARRAY_STREAM:
        callback_func = callback_ndarray;
        break;
    case SNDFILE_STREAM:
        callback_func = callback_sndfile_read;
        ((SndfileData *) user_data)->channel_count = Pa_GetDeviceInfo(out_param->device)->maxOutputChannels;
        out_param->channelCount = Pa_GetDeviceInfo(out_param->device)->maxOutputChannels;
        break;
    case TONE_STREAM:
        callback_func = callback_tone;
        break;
    }

    if (PyObject_HasAttrString(self, "samp_freq")) {
        attr = PyObject_GetAttrString(self, "samp_freq");
        if (attr == NULL) {
            // error, so abort
            return NULL;
        }
        Py_INCREF(attr);
        samp_freq = PyFloat_AsDouble(attr);
        Py_DECREF(attr);
    }
    else {
        return NULL; // Error
    }

    if (in_param != NULL) {
        in_param->suggestedLatency = Pa_GetDeviceInfo(out_param->device)->defaultLowInputLatency;
    }

    if (out_param != NULL) {
        out_param->suggestedLatency = Pa_GetDeviceInfo(out_param->device)->defaultLowOutputLatency;
    }

    err = Pa_OpenStream(&stream,
                        in_param,
                        out_param,
                        samp_freq,
                        paFramesPerBufferUnspecified,
                        paNoFlag,
                        callback_func,
                        user_data);
    ERROR_CHECK;

    return stream;
}