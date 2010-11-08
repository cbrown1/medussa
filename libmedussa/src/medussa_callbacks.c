#include "medussa_callbacks.h"
#include "medussa_matrix.h"

#define TWOPI 6.2831853071795862
#define MAX_FRAME_SIZE 16

int callback_ndarray (const void *pa_buf_in, void *pa_buf_out,
                      unsigned long frames,
                      const PaStreamCallbackTimeInfo *time_info,
                      PaStreamCallbackFlags status_flags,
                      void *user_data)
{
    PyGILState_STATE gstate;

    unsigned int i, j, err, frame_size, cursor, channel_count;

    int loop;        // Boolean
    float *buf_out;  // Points to `pa_buf_out`
    double tmp_buf[MAX_FRAME_SIZE];
    double *mix_mat_arr;
    double *arr_frames;

    PyObject *self, *attr, *out_param;
    PyArrayObject *arr;
    PyArrayObject *mix_mat;

    // Point `buf_out` to actual output buffer
    buf_out = (float *) pa_buf_out;

    // Point `self` to calling instance
    self = (PyObject *) user_data;

    // `PyObject *out_param` from `self.out_param`
    if (PyObject_HasAttrString(self, "out_param")) {
        attr = PyObject_GetAttrString(self, "out_param");
        if (attr == NULL) {
            return -1;
        }
        out_param = attr;
    }
    else {
        return -1;
    }

    // `unsigned int channel_count` from `self.out_param.channelCount`
    if (PyObject_HasAttrString(out_param, "channelCount")) {
        attr = PyObject_GetAttrString(out_param, "channelCount");
        if (attr == NULL) {
            return -1;
        }
        channel_count = (unsigned int) PyInt_AsLong(attr);
    }
    else {
        return -1;
    }

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

    // Point `mix_mat_arr` to data buffer of `mix_mat`
    mix_mat_arr = (double *) PyArray_GETPTR2(mix_mat, 0, 0);

    // Point `arr_frames` to C array of `arr`, move cursor appropriately
    arr_frames = (double *) PyArray_GETPTR2(arr, cursor, 0);

    // Determine `frame_size`, the number of channels, from `arr` (ERROR)
    frame_size = (unsigned int) PyArray_DIM(arr, 1);

    // Copy each frame from of `arr` to the output buffer, multiplying by
    // the mixing matrix each time.
    for (i = 0; i < frames; i++) {
        if (PyArray_DIM(arr, 0) <= cursor+i) {
            break;
        }

        dmatrix_mult(mix_mat_arr, PyArray_DIM(mix_mat, 0), PyArray_DIM(mix_mat, 1),
                     arr_frames+i*frame_size,  frame_size,    1,
                     tmp_buf,     channel_count, 1);
        for (j = 0; j < channel_count; j++){
            buf_out[i*channel_count + j] = (float) tmp_buf[j];
        }
        // */
    }
    cursor += frames;
    //printf("cursor at: %d\n", cursor);

    // Move `self.cursor`
    if (PyObject_HasAttrString(self, "cursor")) {
        gstate = PyGILState_Ensure();
        err = PyObject_SetAttrString(self, "cursor", PyInt_FromLong(cursor));
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

    if (cursor < (PyArray_DIM(arr, 0))) {
        return paContinue;
    }

    // Reset `self.cursor`
    if (PyObject_HasAttrString(self, "cursor")) {
        gstate = PyGILState_Ensure();
        err = PyObject_SetAttrString(self, "cursor", PyInt_FromLong(0));
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

    if (loop) {
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
    PyGILState_STATE gstate;
    float *buf_out;   // Points to `pa_buf_out`

    SNDFILE *fin;
    int frames_read;

    int i, j, err;
    int loop;
    int cursor; // Tracks position in file between callbacks
    int channel_count; // Number of stream output channels
    int frame_size; // Samples per frame for input file
    //double read_buf[1024];
    double *read_buf; // we HAVE to malloc, but we're being ugly in this callback anyway, so oh well
    double tmp_buf[MAX_FRAME_SIZE];
    char *finpath;

    PyObject *self, *attr, *out_param, *finfo;
    PyArrayObject *mix_mat;

    // Point `self` to calling instance
    self = (PyObject *) user_data;

    // `PyObject *out_param` from `self.out_param`
    if (PyObject_HasAttrString(self, "out_param")) {
        attr = PyObject_GetAttrString(self, "out_param");
        if (attr == NULL) {
            return -1;
        }
        out_param = attr;
    }
    else {
        return -1;
    }

    // `unsigned int channel_count` from `self.out_param.channelCount`
    if (PyObject_HasAttrString(out_param, "channelCount")) {
        attr = PyObject_GetAttrString(out_param, "channelCount");
        if (attr == NULL) {
            return -1;
        }
        channel_count = (unsigned int) PyInt_AsLong(attr);
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
        mix_mat = (PyArrayObject *) attr;
    }
    else {
        return -1;
    }

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

    // `char *finpath` from `self.finpath`
    if (PyObject_HasAttrString(self, "finpath")) {
        attr = PyObject_GetAttrString(self, "finpath");
        if (attr == NULL) {
            return -1;
        }
        finpath = PyString_AsString(attr);
    }
    else {
        return -1;
    }

    // `SNDFILE *fin` from `self.fin`
    if (PyObject_HasAttrString(self, "fin")) {
        attr = PyObject_GetAttrString(self, "fin");
        if (attr == NULL) {
            return -1;
        }
        fin = (SNDFILE *) PyInt_AsLong(attr);
    }
    else {
        return -1;
    }

    // `PyObject *finfo` from `self.finfo`
    if (PyObject_HasAttrString(self, "finfo")) {
        attr = PyObject_GetAttrString(self, "finfo");
        if (attr == NULL) {
            return -1;
        }
        finfo = attr;
    }
    else {
        return -1;
    }

    // `int frame_size` from `self.finfo.channels`
    if (PyObject_HasAttrString(finfo, "channels")) {
        attr = PyObject_GetAttrString(finfo, "channels");
        if (attr == NULL) {
            return -1;
        }
        frame_size = (int) PyInt_AsLong(attr);
    }
    else {
        return -1;
    }

    buf_out = (float *) pa_buf_out;

    // This is ugly, but convenient. We can eventually avoid this if really, really necessary
    read_buf = (double *) malloc(1024 * frame_size * sizeof(double));
    frames_read = (int) sf_readf_double (fin, read_buf, frames);

    for (i = 0; i < frames_read; i++) {
        dmatrix_mult((double *) (mix_mat->data), PyArray_DIM(mix_mat, 0), PyArray_DIM(mix_mat, 1), 
                     (read_buf+i*frame_size), frame_size, 1,
                     tmp_buf, channel_count, 1);
        //buf_out[2*i] = (float) read_buf[i];
        //buf_out[2*i+1] = (float) read_buf[i];
        for (j = 0; j < channel_count; j++) {
            buf_out[i*channel_count + j] = (float) tmp_buf[j];
        }
    }
    cursor += frames_read;

    // Move `self.cursor`
    if (PyObject_HasAttrString(self, "cursor")) {
        gstate = PyGILState_Ensure();
        err = PyObject_SetAttrString(self, "cursor", PyInt_FromLong(cursor));
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

    free(read_buf);

    /*
    // Increment time counter by playback delta
    sfd->time += frames;
    /* */

    if (frames_read == frames) {
        // Frames returned equals frames requested, so we didn't reach EOF
        return paContinue;
    }
    else {
        // We've reached EOF
        sf_seek(fin, 0, SEEK_SET); // Reset cursor to start of sound file
        if (loop) {
            return paContinue;
        }
        else {
            // We're really all done
            //sf_close(fin); // Note, this implies we are opening the sndfile each time we call `self.play()`
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

    // `float fs` from `self.fs`
    if (PyObject_HasAttrString(self, "fs")) {
        attr = PyObject_GetAttrString(self, "fs");
        if (attr == NULL) {
            return -1;
        }
        fs = (float) PyFloat_AsDouble(attr);
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
        tone_freq = (float) PyFloat_AsDouble(attr);
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
        t = (unsigned int) PyInt_AsLong(attr);
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
        mix_mat = (PyArrayObject *) attr;
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
        spout = (PaStreamParameters *) PyInt_AsLong(attr);
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
        }
        t++;
    }

    // Set `self.t` to the current time value
    if (PyObject_HasAttrString(self, "t")) {
        attr = PyInt_FromLong(t);
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

    return paContinue;
}
