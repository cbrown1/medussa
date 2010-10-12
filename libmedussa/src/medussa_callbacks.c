#include "medussa_callbacks.h"
#include "medussa_matrix.h"

#define TWOPI 6.2831853071795862

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
    float *scale;

    int i;
    int alpha;
    int beta;
    float *tmp_arr;

    buf_out = (float *) pa_buf_out;

    sfd = (SndfileData *) user_data;
    fin = (SNDFILE *) sfd->fin;
    fin_info = (SF_INFO *) sfd->fin_info;

    tmp_arr = (float *) malloc(sizeof(float) * frames * fin_info->channels);

    frames_read = (int) sf_readf_float (fin, tmp_arr, frames);

    scale = (float *) sfd->scale;



    // Scale the output data buffer now that it's been copied
    //scale = sfd->scale; // Using local var to avoid derefencing pointer in loop
    //for (i = 0; i < (frames_read * fin_info->channels); i++) {
    //    tmp_arr[i] = tmp_arr[i] * scale;
    //}

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
    unsigned int i, j, t, frame_size;
    float *buf_out;
    
    float fs, tone_freq;

    float *mix_mat_arr;
    int err;

    PaStreamParameters *spout;

    PyObject *self, *attr;
    PyArrayObject *mix_mat;


    // Point `self` to calling instance
    self = (PyObject *) user_data;
    Py_INCREF(self);
    
    // `float fs` from `self.fs`
    if (PyObject_HasAttrString(self, "fs")) {
        attr = PyObject_GetAttrString(self, "fs");
        if (attr == NULL) {
            return -1;
        }
        Py_INCREF(attr);
        fs = (float) PyFloat_AsDouble(attr);
        Py_DECREF(attr);
    }
    else {
        return -1;
    }

    // `float tone_freq` from `self.tone_freq`
    if (PyObject_HasAttrString(self, "tone_freq")) {
        attr = PyObject_GetAttrString(self, "tone_freq");
        if (attr == NULL) {
            return -1;
        }
        Py_INCREF(attr);
        tone_freq = (float) PyFloat_AsDouble(attr);
        Py_DECREF(attr);
    }
    else {
        return -1;
    }

    // `unsigned int t` from `self.t`
    if (PyObject_HasAttrString(self, "t")) {
        attr = PyObject_GetAttrString(self, "t");
        if (attr == NULL) {
            return -1;
        }
        Py_INCREF(attr);
        t = (unsigned int) PyInt_AsLong(attr);
        Py_DECREF(attr);
    }
    else {
        return -1;
    }

    // `PyArrayObject *arr` from `self.mix_mat`
    if (PyObject_HasAttrString(self, "mix_mat")) {
        attr = PyObject_GetAttrString(self, "mix_mat");
        if (attr == NULL) {
            return -1;
        }
        Py_INCREF(attr);
        mix_mat = (PyArrayObject *) attr;
        Py_INCREF(mix_mat);
        Py_DECREF(attr);
    }
    else {
        return -1;
    }

    // `PaStreamParameters *spout` from `Stream.out_param`
    if (PyObject_HasAttrString(self, "out_param")) {
        attr = PyObject_GetAttrString(self, "out_param");
        if (attr == NULL) {
            return -1;
        }
        Py_INCREF(attr);
        spout = (PaStreamParameters *) PyInt_AsLong(attr);
        Py_DECREF(attr);
    }
    else {
        return -1;
    }

    frame_size = spout->channelCount;

    // Point to data array of `mix_mat`
    mix_mat_arr = (float *) PyArray_DATA(mix_mat);

    // Point to actual output buffer
    buf_out = (float *) pa_buf_out;
    
    // Main loop for tone generation
    for (i = 0; i < frames; i++) {
        for (j = 0; j < frame_size; j++) {
            // Note that we implicitly assume `mix_mat` is an `n x 1` matrix
            buf_out[i*frame_size + j] = (float) (sin(TWOPI * t / fs * tone_freq) * mix_mat_arr[j]);
        }
        t++;
    }

    // Set `self.t` to the current time value
    if (PyObject_HasAttrString(self, "t")) {
        err = PyObject_SetAttrString(self, "t", PyLong_FromUnsignedLong(t));
        if (err == -1) {
            return -1;
        }
    }
    else {
        return -1;
    }


    Py_DECREF(mix_mat);
    Py_DECREF(self);

    return paContinue;
}
