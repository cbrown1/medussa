#include "medussa_callbacks.h"
#include "medussa_matrix.h"

#define TWOPI 6.2831853071795862
#define MAX_FRAME_SIZE 16


void execute_stream_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data )
{
    stream_user_data *sud = (stream_user_data *) data;
    stream_command resultCommand;

    switch( command->command ){
    case STREAM_COMMAND_SET_MATRICES:

        /* post old matrices back to python to be freed */
        resultCommand.command = STREAM_COMMAND_FREE_MATRICES;
        resultCommand.data_ptr0 = sud->mix_mat;
        resultCommand.data_ptr1 = sud->mute_mat;
        PaUtil_WriteRingBuffer(resultQueue, &resultCommand, 1 );

        /* install new matrices */
        sud->mix_mat = (medussa_dmatrix*)command->data_ptr0;
        sud->mute_mat = (medussa_dmatrix*)command->data_ptr1;

        break;

    case STREAM_COMMAND_SET_IS_MUTED:
        sud->is_muted = command->data_uint;
        break;

    }
}

void execute_finite_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data )
{
    finite_user_data *fud = (finite_user_data *) data;

    switch( command->command ){

    default:
        execute_stream_user_data_command( resultQueue, command, fud->parent );
    }
}

void execute_array_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data )
{
    array_user_data *aud = (array_user_data *) data;

    switch( command->command ){

    default:
        execute_finite_user_data_command( resultQueue, command, aud->parent );
    }
}

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
    medussa_dmatrix *mix_mat;
    double *arr;
    
    stream_user_data *sud;
    finite_user_data *fud;
    array_user_data *aud;

    // Point `buf_out` to actual output buffer
    buf_out = (float *) pa_buf_out;

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
    sud = (stream_user_data *) fud->parent;
    if (aud == NULL) {
        printf("ERROR [callback_ndarray]: `stream_user_data` is NULL\n");
        return 1;
    }

    execute_commands_in_pa_callback( sud->command_queues, execute_array_user_data_command, aud );

    loop = fud->loop;

    mix_mat = sud->is_muted ? sud->mute_mat : sud->mix_mat;

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
        
        dmatrix_mult(mix_mat->mat, mix_mat->mat_0, mix_mat->mat_1,
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


void execute_sndfile_read_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data )
{
    sndfile_user_data *sfud = (sndfile_user_data *) data;

    switch( command->command ){

    default:
        execute_finite_user_data_command( resultQueue, command, sfud->parent );
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
    medussa_dmatrix *mix_mat;

    sfud = (sndfile_user_data *) user_data;
    fud  = (finite_user_data *)  sfud->parent;
    stud = (stream_user_data *)  fud->parent;

    execute_commands_in_pa_callback( stud->command_queues, execute_sndfile_read_user_data_command, sfud );

    // Begin attribute acquisition
    out_param = stud->out_param;
    finfo = sfud->finfo;
    channel_count = out_param->channelCount;
    mix_mat = stud->is_muted ? stud->mute_mat : stud->mix_mat;
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
        dmatrix_mult(mix_mat->mat, mix_mat->mat_0, mix_mat->mat_1,
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


void execute_tone_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data )
{
    tone_user_data *tud = (tone_user_data *) data;

    switch( command->command ){

    default:
        execute_stream_user_data_command( resultQueue, command, tud->parent );
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

    medussa_dmatrix *mix_mat;

    PaStreamParameters *spout;

    stream_user_data *sud;
    tone_user_data *tud;
    tud = (tone_user_data *) user_data;
    sud = (stream_user_data *) tud->parent;

    execute_commands_in_pa_callback( sud->command_queues, execute_tone_user_data_command, tud );

    // Point `self` to calling instance
    // `float fs` from `self.fs`
    fs = (float) sud->fs;

    // `float tone_freq` from `self.tone_freq`
    tone_freq = (float) tud->tone_freq;

    // `unsigned int t` from `self.t`
    t = tud->t;

    // `PaStreamParameters *spout` from `Stream.out_param`
    spout = (PaStreamParameters *) sud->out_param;

    mix_mat = sud->is_muted ? sud->mute_mat : sud->mix_mat;

    // Point to actual output buffer
    buf_out = (float *) pa_buf_out;

    frame_size = mix_mat->mat_0;

    //printf("%f, %f, %d\n", fs, tone_freq, frame_size);
    

    // Main loop for tone generation
    for (i = 0; i < frames; i++) {
        for (j = 0; j < frame_size; j++) {
            // Note that we implicitly assume `mix_mat` is an `n x 1` matrix
            buf_out[i*frame_size + j] = (float) (sin(TWOPI * ((float) t) / fs * tone_freq) * ((float) mix_mat->mat[j]));
        }
        t++;
    }
   
    // Set `self.t` to the current time value
    tud->t = t;

    return paContinue;
}


void execute_white_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data )
{
    white_user_data *wud = (white_user_data *) data;

    switch( command->command ){

    default:
        execute_stream_user_data_command( resultQueue, command, wud->parent );
    }
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

    medussa_dmatrix *mix_mat;

    double tmp;

    PaStreamParameters *spout;

    white_user_data *wud;
    stream_user_data *sud;

    // PRNG variables
    rk_state *rks;

    wud = (white_user_data *) user_data;
    sud = (stream_user_data *) wud->parent;

    execute_commands_in_pa_callback( sud->command_queues, execute_stream_user_data_command, sud );

    fs = sud->fs;
    rks = wud->rks;
    mix_mat = sud->is_muted ? sud->mute_mat : sud->mix_mat;
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
            buf_out[i*frame_size + j] = tmp * ((float) mix_mat->mat[j]);
        }
    }

    return paContinue;
}


void execute_pink_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data )
{
    pink_user_data *pud = (pink_user_data *) data;

    switch( command->command ){

    default:
        execute_stream_user_data_command( resultQueue, command, pud->parent );
    }
}

int callback_pink  (const void *pa_buf_in, void *pa_buf_out,
                    unsigned long frames,
                    const PaStreamCallbackTimeInfo *time_info,
                    PaStreamCallbackFlags status_flags,
                    void *user_data)
{
    unsigned int i, j, frame_size;
    float *buf_out;
    
    float fs;

    medussa_dmatrix *mix_mat;

    double tmp;

    PaStreamParameters *spout;

    pink_user_data *pud;
    stream_user_data *sud;

    pink_noise_t *pn;

    pud = (pink_user_data *) user_data;
    sud = (stream_user_data *) pud->parent;

    execute_commands_in_pa_callback( sud->command_queues, execute_stream_user_data_command, sud );

    fs = sud->fs;
    pn = (pink_noise_t *) pud->pn;

    mix_mat = sud->is_muted ? sud->mute_mat : sud->mix_mat;
    spout = sud->out_param;

    buf_out = (float *) pa_buf_out;

    frame_size = spout->channelCount;

    // Main loop for tone generation
    for (i = 0; i < frames; i++) {
        for (j = 0; j < frame_size; j++) {
            // Note that we implicitly assume `mix_mat` is an `n x 1` matrix
            tmp = generate_pink_noise_sample(pn);
            //printf("%.6f\n", rk_gauss(state));
            if (tmp < -1.0) {
                tmp = -1.0;
                //printf("DEBUG: clipped below\n");
            }
            if (tmp > 1.0) {
                tmp = 1.0;
                //printf("DEBUG: clipped above\n");
            }
            buf_out[i*frame_size + j] = tmp * ((float) mix_mat->mat[j]);
        }
    }

    return paContinue;
}

