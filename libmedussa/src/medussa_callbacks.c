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

    printf("DEBUG-NDARRAY-1: start of callback\n");
    printf("DEBUG-NDARRAY: value of `frames` == %u\n", frames);

    // Point `buf_out` to actual output buffer
    buf_out = (float *) pa_buf_out;

    printf("DEBUG-NDARRAY-2\n");

    // Point `self` to calling instance
    aud = (array_user_data *) user_data;
    if (aud == NULL) {
        printf("ERROR [callback_ndarray]: `array_user_data` is NULL\n");
        return 1;
    }
    fud = (finite_user_data *) aud->parent;
    if (aud == NULL) {
        printf("ERROR [callback_ndarray]: `finite_user_data` is NULL\n");
        return 1;
    }
    printf("DEBUG-NDARRAY-2-0: value of `fud` == %p\n", fud);
    sud = (stream_user_data *) fud->parent;
    if (aud == NULL) {
        printf("ERROR [callback_ndarray]: `stream_user_data` is NULL\n");
        return 1;
    }

    printf("DEBUG-NDARRAY-3\n");

    loop = fud->loop;

    printf("DEBUG-NDARRAY-4\n");

    // Point `mix_mat_arr` to data buffer of `mix_mat`
    mix_mat = (double *) sud->mix_mat;

    printf("DEBUG-NDARRAY-5\n");

    // Determine `frame_size`, the number of channels, from `arr` (ERROR)
    frame_size = (unsigned int) aud->ndarr_1;
    printf("DEBUG-NDARRAY: `frame_size` == %d\n", frame_size);

    printf("DEBUG-NDARRAY-6\n");

    // Point `arr_frames` to C array of `arr`, move cursor appropriately
    cursor = fud->cursor;
    arr = aud->ndarr + cursor*frame_size;

    printf("DEBUG-NDARRAY-7\n");

    channel_count = sud->out_param->channelCount;

    printf("DEBUG-NDARRAY-8: before main loop\n");

    printf("\tDEBUG-NDARRAY-8-0: value of `fud`: %p\n", fud);
    printf("\tDEBUG-NDARRAY-8-1: value of `cursor`: %d\n", cursor);

    // Copy each frame from of `arr` to the output buffer, multiplying by
    // the mixing matrix each time.
    for (i = 0; i < frames; i++) {
        if (aud->ndarr_0 <= (fud->cursor+i)) {
            break;
        }
//        printf("\tDEBUG-NDARRAY-8-in_loop-0: value of `fud`: %p\n", fud);
//        printf("\t\tDEBUG: sud->mix_mat_0: %d\n", sud->mix_mat_0);
//        printf("\t\tDEBUG: sud->mix_mat_1: %d\n", sud->mix_mat_1);
        
        dmatrix_mult(mix_mat,
                     sud->mix_mat_0, sud->mix_mat_1,
                     arr+i*frame_size,
                     frame_size, 1,
                     tmp_buf,
                     channel_count, 1);
//        printf("\tDEBUG-NDARRAY-8-in_loop-1: value of `fud`: %p\n", fud);

        for (j = 0; j < channel_count; j++) {
            buf_out[i*channel_count + j] = (float) tmp_buf[j];
        }
//        printf("\tDEBUG-NDARRAY-8-in_loop-2: value of `fud`: %p\n", fud);
    }
    cursor += i;

    printf("DEBUG-NDARRAY-9: after main loop\n");

    printf("\tDEBUG: value of `cursor`: %d\n", cursor);
    printf("\tDEBUG: value of `fud`: %p\n", fud);
    printf("\tDEBUG: value of `fud->cursor`: %d\n", fud->cursor);

    // Move `self.cursor`
    fud->cursor = cursor;

    printf("DEBUG-NDARRAY-10\n");

    if (cursor < aud->ndarr_0) {
        return paContinue;
    }

    printf("DEBUG-NDARRAY-11\n");

    // Reset `self.cursor`
    fud->cursor = 0;

    printf("DEBUG-NDARRAY-12\n");

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

    double *mix_mat;

    double tmp;

    PaStreamParameters *spout;

    white_user_data *wud;
    stream_user_data *sud;

    // PRNG variables
    rk_state *rks;

    wud = (white_user_data *) user_data;
    sud = (stream_user_data *) wud->parent;

    fs = sud->fs;
    rks = wud->rks;
    mix_mat = (double *) sud->mix_mat;
    spout = sud->out_param;

    buf_out = (float *) pa_buf_out;

    frame_size = spout->channelCount;

    // Main loop for tone generation
    for (i = 0; i < frames; i++) {
        for (j = 0; j < frame_size; j++) {
            // Note that we implicitly assume `mix_mat` is an `n x 1` matrix
            tmp = rk_gauss(rks) * 0.1;
            //printf("%.6f\n", rk_gauss(state));
            if (tmp < -1.0) {
                tmp = -1.0;
                //printf("DEBUG: clipped below\n");
            }
            if (tmp > 1.0) {
                tmp = 1.0;
                //printf("DEBUG: clipped above\n");
            }
            buf_out[i*frame_size + j] = tmp * ((float) mix_mat[j]);
        }
    }

    return paContinue;
}
