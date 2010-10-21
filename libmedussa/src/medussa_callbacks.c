#include "medussa_callbacks.h"
#include "medussa_matrix.h"

#define TWOPI 6.2831853071795862

int callback_ndarray (const void *pa_buf_in, void *pa_buf_out,
                      unsigned long frames,
                      const PaStreamCallbackTimeInfo *time_info,
                      PaStreamCallbackFlags status_flags,
                      void *user_data)
{
    PyGILState_STATE gstate;

    unsigned int i, j, err, frame_size, cursor;

    int loop;        // Boolean
    float *buf_out;  // Points to `pa_buf_out`
    double *mix_mat_arr;
    double *arr_frames;

    PyObject *self, *attr;
    PyArrayObject *arr;
    PyArrayObject *mix_mat;

    // Point `buf_out` to actual output buffer
    buf_out = (float *) pa_buf_out;

    // Point `self` to calling instance
    self = (PyObject *) user_data;
    //PyObject_Print(self, stdout, Py_PRINT_RAW);

    //printf("DEBUG 1\n");

    // `PyArrayObject *arr` from `self.arr`
    if (PyObject_HasAttrString(self, "arr")) {
        attr = PyObject_GetAttrString(self, "arr");
        if (attr == NULL) {
            return -1;
        }
        arr = (PyArrayObject *) attr;
    }
    else {
        return -1;
    }

    //printf("DEBUG 2\n");

    // `PyArrayObject *mix_mat` from `self.mix_mat`
    if (PyObject_HasAttrString(self, "mix_mat")) {
        attr = PyObject_GetAttrString(self, "mix_mat");
        if (attr == NULL) {
            return -1;
        }
        mix_mat = (PyArrayObject *) attr;
    }
    else {
        return -1;
    }

    //printf("DEBUG 3\n");

    // `int loop` from `self.loop`
    if (PyObject_HasAttrString(self, "loop")) {
        attr = PyObject_GetAttrString(self, "loop");
        if (attr == NULL) {
            return -1;
        }
        loop = (int) PyInt_AsLong(attr);
    }
    else {
        return -1;
    }

    //printf("DEBUG 4\n");

    // `unsigned int cursor` from `self.cursor`
    if (PyObject_HasAttrString(self, "cursor")) {
        attr = PyObject_GetAttrString(self, "cursor");
        if (attr == NULL) {
            return -1;
        }
        cursor = (unsigned int) PyInt_AsLong(attr);
    }
    else {
        return -1;
    }

    //printf("DEBUG 5\n");

    // Point `mix_mat_arr` to data buffer of `mix_mat`
    mix_mat_arr = (double *) mix_mat->data;

    // Point `arr_frames` to C array of `arr`
    arr_frames = (double *) arr->data;

    // Determine `frame_size`, the number of channels, from `arr`
    frame_size = (unsigned int) PyArray_DIM(arr, 1);

    //printf("cursor WAS at: %u\n", cursor);
    // Copy each frame from of `arr` to the output buffer, multiplying by
    // the mixing matrix each time.
    for (i = 0; i < frames; i++) {
        if (PyArray_DIM(arr, 0) <= (cursor+i)) {
            break;
        }
        for (j = 0; j < frame_size; j++) {
            // printf("%f ", (float) *((double *) PyArray_GETPTR2(arr, i, j)));
            // buf_out[i*frame_size + j] = (float) arr_frames[i*frame_size + j];
            //printf("%d %d\n", i, j);
            buf_out[i*frame_size + j] = (float) *((double *) PyArray_GETPTR2(arr, (cursor+i), j));
            //buf_out[i*frame_size + j] = (float) (sin(TWOPI * ((float) i) / 44100.0 * 400.0));
        }
        //printf("one frame\n");
    }
    cursor += frames;
    //printf("cursor is at: %u\n", cursor);

    //printf("DEBUG 6\n");

    // Move `self.cursor`
    if (PyObject_HasAttrString(self, "cursor")) { // python bug?
        //printf("DEBUG 6-0\n");
        //attr = PyInt_FromLong(cursor);
        //Py_INCREF(attr);
        gstate = PyGILState_Ensure();
        //printf("DEBUG 6-1\n");
        err = PyObject_SetAttrString(self, "cursor", PyInt_FromLong(cursor));
        //printf("DEBUG 6-2\n");
        PyGILState_Release(gstate);
        if (err == -1) {
            printf("DEBUG: ERROR\n");
            return -1;
        }
    }
    else {
        printf("ERROR: no `cursor` attribute\n");
        return -1;
    }

    //printf("DEBUG 7\n");

    if (cursor < PyArray_DIM(arr, 0)) {
        //printf("not done, continuing...\n");
        return paContinue;
    }

    //printf("DEBUG 8\n");

    if (loop) {
        //printf("looping, so continuing...\n");
        return paContinue;
    }
    else {
        //printf("array is played\n");
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

    double *mix_mat_arr;
    int err;

    PyGILState_STATE gstate;

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

    // `PyArrayObject *mix_mat` from `self.mix_mat`
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
    if (PyObject_HasAttrString(self, "spout_ptr")) {
        attr = PyObject_GetAttrString(self, "spout_ptr");
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
    mix_mat_arr = (double *) PyArray_DATA(mix_mat);

    // Point to actual output buffer
    buf_out = (float *) pa_buf_out;
    
    // Main loop for tone generation
    for (i = 0; i < frames; i++) {
        for (j = 0; j < frame_size; j++) {
            // Note that we implicitly assume `mix_mat` is an `n x 1` matrix
            buf_out[i*frame_size + j] = (float) (sin(TWOPI * ((float) t) / fs * tone_freq) * ((float) mix_mat_arr[j]));
            // printf("%f\n", (float) (sin(TWOPI * t / fs * tone_freq) * mix_mat_arr[j]));
        }
        t++;
    }

    // Set `self.t` to the current time value
    if (PyObject_HasAttrString(self, "t")) {
        attr = PyInt_FromLong(t);
        Py_INCREF(attr);
        gstate = PyGILState_Ensure();
        err = PyObject_SetAttrString(self, "t", attr);
        PyGILState_Release(gstate);
        if (err == -1) {
            return -1;
        }
    }
    else {
        printf("ERROR: no `t` attribute\n");
        return -1;
    }

    Py_DECREF(mix_mat);
    Py_DECREF(self);

    return paContinue;
}
