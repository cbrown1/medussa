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

    frames_read = sf_readf_float (fin, tmp_arr, frames);

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
    unsigned int t;
    float *buf_out;
    
    float fs, tone_freq;

    float *mix_mat;
    int i, m, n;

    PyObject *self, *attr;
    PyArrayObject *arr;

    // Point `self` to calling instance
    self = (PyObject *) user_data;
    Py_INCREF(self);
    
    // `float fs` from `self.fs`
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


    // Base vector which is then mixed to the desired output channels
    // via a matrix multiplication.
    float *vin;

    buf_out = (float *) pa_buf_out;

    // Temporary (bad due to possible interrupt priority)
    vin = (float *) malloc(sizeof(float) * frames); 
    
    // Generate 1 channel worth of unmixed samples
    for (i = 0; i < frames; i++) {
        vin[i] = sin(TWOPI * t / fs * tone_freq);
        t++;
    }

    // Mix samples to properly strided output matrix
    //matrix_mult(mix_mat, m, n,
                

    

    free(vin);

    Py_DECREF(self);

    return paContinue;
}
