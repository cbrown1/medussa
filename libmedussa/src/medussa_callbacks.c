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
    unsigned int i, j, frame_size, cursor, channel_count;

    int loop;        // Boolean
    float *buf_out;  // Points to `pa_buf_out`
    double tmp_buf[MAX_FRAME_SIZE];
    double *mix_mat;
    double *arr;
    
    stream_user_data *sud;
    finite_user_data *fud;
    array_user_data *aud;

    // Point `buf_out` to actual output buffer
    buf_out = (float *) pa_buf_out;

    // Point `self` to calling instance
    aud = (array_user_data *) user_data;
    fud = (finite_user_data *) aud->parent;
    sud = (stream_user_data *) fud->parent;

    loop = fud->loop;

    // Point `mix_mat_arr` to data buffer of `mix_mat`
    mix_mat = (double *) sud->mix_mat;

    // Determine `frame_size`, the number of channels, from `arr` (ERROR)
    frame_size = (unsigned int) aud->ndarr_1;

    // Point `arr_frames` to C array of `arr`, move cursor appropriately
    cursor = fud->cursor;
    arr = aud->ndarr + cursor*frame_size;

    channel_count = sud->out_param->channelCount;

    // Copy each frame from of `arr` to the output buffer, multiplying by
    // the mixing matrix each time.
    for (i = 0; i < frames; i++) {
        if (aud->ndarr_0 <= (fud->cursor+i)) {
            break;
        }
        
        dmatrix_mult(mix_mat,
                     sud->mix_mat_0, sud->mix_mat_1,
                     arr+i*frame_size,
                     frame_size, 1,
                     tmp_buf,
                     channel_count, 1);

        for (j = 0; j < channel_count; j++) {
            buf_out[i*channel_count + j] = (float) tmp_buf[j];
        }
    }
    cursor += i;

    // Move `self.cursor`
    fud->cursor = cursor;

    if (cursor < aud->ndarr_0) {
        return paContinue;
    }

    // Reset `self.cursor`
    fud->cursor = 0;

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
    float *buf_out;   // Points to `pa_buf_out`

    SNDFILE *fin;
    int frames_read;

    int i, j;
    int loop;
    int cursor; // Tracks position in file between callbacks
    int channel_count; // Number of stream output channels
    int frame_size; // Samples per frame for input file
    //double read_buf[1024];
    double *read_buf; // we HAVE to malloc, but we're being ugly in this callback anyway, so oh well
    double tmp_buf[MAX_FRAME_SIZE];
    char *finpath;

    sndfile_user_data *sfud;
    finite_user_data  *fud;
    stream_user_data  *stud;
        
    SF_INFO *finfo;
    PaStreamParameters *out_param;
    double *mix_mat;

    sfud = (sndfile_user_data *) user_data;
    fud  = (finite_user_data *)  sfud->parent;
    stud = (stream_user_data *)  fud->parent;


    // Begin attribute acquisition
    out_param = stud->out_param;
    finfo = sfud->finfo;
    channel_count = out_param->channelCount;
    mix_mat = (double *) stud->mix_mat;
    loop = fud->loop;
    cursor = fud->cursor;
    finpath = sfud->finpath;
    fin = sfud->fin;
    frame_size = finfo->channels;
    // End attribute acquisition


    buf_out = (float *) pa_buf_out;
    if (buf_out == NULL) { printf("DEBUG 1: NULL pointer\n"); }

    sf_seek(fin, cursor, SEEK_SET);

    // This is ugly, but convenient. We can eventually avoid this if really, really necessary
    read_buf = (double *) malloc(1024 * frame_size * sizeof(double));
    if (read_buf == NULL) { printf("DEBUG 2: NULL pointer\n"); }

    frames_read = (int) sf_readf_double (fin, read_buf, frames);

    for (i = 0; i < frames_read; i++) {
        dmatrix_mult(mix_mat, stud->mix_mat_0, stud->mix_mat_1, 
                     (read_buf+i*frame_size), frame_size, 1,
                     tmp_buf, channel_count, 1);
        for (j = 0; j < channel_count; j++) {
            buf_out[i*channel_count + j] = (float) tmp_buf[j];
        }
    }
    cursor += frames_read;

    // Move `self.cursor`
    fud->cursor = cursor;

    if (read_buf == NULL) { printf("DEBUG 3: NULL pointer\n"); }
    free(read_buf);

    if (frames_read == frames) {
        // Frames returned equals frames requested, so we didn't reach EOF
        return paContinue;
    }
    else {
        // We've reached EOF
        sf_seek(fin, 0, SEEK_SET); // Reset `libsndfile` cursor to start of sound file

        // Move `self.cursor`
        fud->cursor = 0;

        if (loop) {
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

    double *mix_mat;

    PaStreamParameters *spout;

    stream_user_data *sud;
    tone_user_data *tud;
    tud = (tone_user_data *) user_data;
    sud = (stream_user_data *) tud->parent;

    // Point `self` to calling instance
    // `float fs` from `self.fs`
    fs = (float) sud->fs;

    // `float tone_freq` from `self.tone_freq`
    tone_freq = (float) tud->tone_freq;

    // `unsigned int t` from `self.t`
    t = tud->t;

    // `PaStreamParameters *spout` from `Stream.out_param`
    spout = (PaStreamParameters *) sud->out_param;

    // Point to data array of `mix_mat`
    mix_mat = (double *) sud->mix_mat;

    // Point to actual output buffer
    buf_out = (float *) pa_buf_out;

    frame_size = sud->mix_mat_0;

    //printf("%f, %f, %d\n", fs, tone_freq, frame_size);
    

    // Main loop for tone generation
    for (i = 0; i < frames; i++) {
        for (j = 0; j < frame_size; j++) {
            // Note that we implicitly assume `mix_mat` is an `n x 1` matrix
            buf_out[i*frame_size + j] = (float) (sin(TWOPI * ((float) t) / fs * tone_freq) * ((float) mix_mat[j]));
        }
        t++;
    }
   
    // Set `self.t` to the current time value
    tud->t = t;

    return paContinue;
}

int callback_white  (const void *pa_buf_in, void *pa_buf_out,
                     unsigned long frames,
                     const PaStreamCallbackTimeInfo *time_info,
                     PaStreamCallbackFlags status_flags,
                     void *user_data)
{
    unsigned int i, j, frame_size;
    float *buf_out;
    
    float fs;

    double *mix_mat_arr;

    double tmp;

    PyGILState_STATE gstate;

    PaStreamParameters *spout;

    PyObject *self, *attr;
    PyArrayObject *mix_mat;

    // PRNG variables
    rk_state *state;

    // Point `self` to calling instance
    self = (PyObject *) user_data;

    gstate = PyGILState_Ensure();

    // `float fs` from `self.fs`
    if (PyObject_HasAttrString(self, "fs")) {
        attr = PyObject_GetAttrString(self, "fs");
        if (attr == NULL) {
            return -1;
        }
        fs = (float) PyFloat_AsDouble(attr);
        Py_CLEAR(attr);
    }
    else {
        return -1;
    }

    // `rk_state *state` from `self.rk_state_ptr`
    if (PyObject_HasAttrString(self, "rk_state_ptr")) {
        attr = PyObject_GetAttrString(self, "rk_state_ptr");
        if (attr == NULL) {
            return -1;
        }
        state = (rk_state *) PyInt_AsLong(attr);
        Py_CLEAR(attr);
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
        Py_CLEAR(attr);
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
        Py_CLEAR(attr);
    }
    else {
        return -1;
    }
    frame_size = spout->channelCount;

    PyGILState_Release(gstate);


    // Point to data array of `mix_mat`
    mix_mat_arr = (double *) PyArray_DATA(mix_mat);

    // Point to actual output buffer
    buf_out = (float *) pa_buf_out;
    

    // Main loop for tone generation
    for (i = 0; i < frames; i++) {
        for (j = 0; j < frame_size; j++) {
            // Note that we implicitly assume `mix_mat` is an `n x 1` matrix
            tmp = rk_gauss(state) * 0.1;
            //printf("%.6f\n", rk_gauss(state));
            if (tmp < -1.0) {
                tmp = -1.0;
                //printf("DEBUG: clipped below\n");
            }
            if (tmp > 1.0) {
                tmp = 1.0;
                //printf("DEBUG: clipped above\n");
            }
            buf_out[i*frame_size + j] = tmp * ((float) mix_mat_arr[j]);
        }
    }

    return paContinue;
}
