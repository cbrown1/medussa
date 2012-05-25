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
    case FINITE_STREAM_COMMAND_SET_CURSOR:

        fud->cursor = command->data_uint;

        /* TODO: when we implement async streaming from a separate thread, we may do something different here for the soundfile stream */

        break;

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
                      unsigned long frame_count,
                      const PaStreamCallbackTimeInfo *time_info,
                      PaStreamCallbackFlags status_flags,
                      void *user_data)
{
    unsigned int i, j, array_channel_count, stream_channel_count;

    int loop;        // Boolean
    float *buf_out;  // Points to `pa_buf_out`
    double tmp_buf[MAX_FRAME_SIZE];
    medussa_dmatrix *mix_mat;
    double *arr;
    
    stream_user_data *sud;
    finite_user_data *fud;
    array_user_data *aud;

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

    stream_channel_count = sud->out_param->channelCount;

    assert( mix_mat->mat_0 == stream_channel_count ); // matrix must have as many output channels as our stream
    assert( mix_mat->mat_1 == file_channel_count ); // matrix must have same number of source channels as the file


    // Determine `array_channel_count`, the number of channels, from `arr`
    array_channel_count = (unsigned int) aud->ndarr_1;

    // Point `arr_frames` to C array of `arr`, move cursor appropriately
    arr = aud->ndarr + fud->cursor*array_channel_count;

    // Copy each frame from of `arr` to the output buffer, multiplying by
    // the mixing matrix each time.
    buf_out = (float *) pa_buf_out;
    for (i = 0; i < frame_count; i++) {
        if ( fud->cursor+i >= (unsigned)aud->ndarr_0 ) {
            break;
        }
        
        dmatrix_mult(mix_mat->mat, mix_mat->mat_0, mix_mat->mat_1,
                     arr+i*array_channel_count,
                     array_channel_count, 1,
                     tmp_buf,
                     stream_channel_count, 1);

        for (j = 0; j < stream_channel_count; j++) {
            buf_out[i*stream_channel_count + j] = (float) tmp_buf[j];
        }
    }

    // if we're at the end of the array write silence into the remainder of the output buffer
    for (; i < frame_count; i++) {
         for (j = 0; j < stream_channel_count; j++) {
                buf_out[i*stream_channel_count + j] = 0.0f;
         }
    }

    // Move `self.cursor`
    fud->cursor = (fud->cursor + i); // Assume ATOMIC STORE

    if (fud->cursor < (unsigned int)aud->ndarr_0) {
        return paContinue;
    }

    if (loop) {
        fud->cursor = 0;
        return paContinue;
    }
    else {
        // NOTE: if the stream has completed we don't reset the cursor to zero.
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
                           unsigned long frame_count,
                           const PaStreamCallbackTimeInfo *time_info,
                           PaStreamCallbackFlags status_flags,
                           void *user_data)
{
    float *buf_out;   // Points to `pa_buf_out`
    SNDFILE *fin;
    int frames_read;
    int i, j;
    int loop;
    int stream_channel_count; // Number of stream output channels
    int file_channel_count; // Samples per frame for input file
    //double read_buf[1024];
    double *read_buf; // we HAVE to malloc, but we're being ugly in this callback anyway, so oh well
    double tmp_buf[MAX_FRAME_SIZE];
    char *finpath;
    SF_INFO *finfo;
    medussa_dmatrix *mix_mat;
    sndfile_user_data *sfud;
    finite_user_data  *fud;
    stream_user_data  *stud;
    
    sfud = (sndfile_user_data *) user_data;
    fud  = (finite_user_data *)  sfud->parent;
    stud = (stream_user_data *)  fud->parent;

    execute_commands_in_pa_callback( stud->command_queues, execute_sndfile_read_user_data_command, sfud );

    // Begin attribute acquisition
    finfo = sfud->finfo;
    stream_channel_count = stud->out_param->channelCount;
    mix_mat = stud->is_muted ? stud->mute_mat : stud->mix_mat;
    loop = fud->loop;
    finpath = sfud->finpath;
    fin = sfud->fin;
    file_channel_count = finfo->channels;
    // End attribute acquisition

    assert( mix_mat->mat_0 == stream_channel_count ); // matrix must have as many output channels as our stream
    assert( mix_mat->mat_1 == file_channel_count ); // matrix must have same number of source channels as the file

    // read from the file
    sf_seek(fin, fud->cursor, SEEK_SET);

    // This is ugly, but convenient. We can eventually avoid this if really, really necessary [yes it is really really necessary to not call malloc in a callback. --rossb]
    read_buf = (double *) malloc(1024 * file_channel_count * sizeof(double));
    if (read_buf == NULL) { printf("DEBUG 2: NULL pointer\n"); }

    frames_read = (int) sf_readf_double (fin, read_buf, frame_count);
    //

    buf_out = (float *) pa_buf_out;
    if (buf_out == NULL) { printf("DEBUG 1: NULL pointer\n"); }
    for (i = 0; i < frames_read; i++) {
        dmatrix_mult(mix_mat->mat, mix_mat->mat_0, mix_mat->mat_1,
                     (read_buf+i*file_channel_count), file_channel_count, 1,
                     tmp_buf, stream_channel_count, 1);
        for (j = 0; j < stream_channel_count; j++) {
            buf_out[i*stream_channel_count + j] = (float) tmp_buf[j];
        }
    }

    // Move `self.cursor`
    fud->cursor = (fud->cursor + frames_read);  // Assume ATOMIC STORE

    if (read_buf == NULL) { printf("DEBUG 3: NULL pointer\n"); }
    free(read_buf);

    if (frames_read == frame_count) {
        // Frames returned equals frames requested, so we didn't reach EOF
        return paContinue;
    }
    else {
        // write silence into the remainder of the output buffer
        for( i = frames_read; i < (int)frame_count; ++i ) {
            for (j = 0; j < stream_channel_count; j++) {
                buf_out[i*stream_channel_count + j] = 0.0f;
            }
        }

        // We've reached EOF
        sf_seek(fin, 0, SEEK_SET); // Reset `libsndfile` cursor to start of sound file

        if (loop) {
            fud->cursor = 0;
            return paContinue;
        }
        else {
            // We're really all done
            // NOTE: if the stream has completed we don't reset the cursor to zero.
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
                    unsigned long frame_count,
                    const PaStreamCallbackTimeInfo *time_info,
                    PaStreamCallbackFlags status_flags,
                    void *user_data)
{
    unsigned int i, j, t, channel_count;
    float *buf_out;
    float fs, tone_freq;
    medussa_dmatrix *mix_mat;
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
    mix_mat = sud->is_muted ? sud->mute_mat : sud->mix_mat;

    channel_count = sud->out_param->channelCount;
    assert( mix_mat->mat_0 == channel_count ); // matrix must have as many output channels as our stream
    assert( mix_mat->mat_1 == 1 ); // matrix must have 1 source channel: our tone generator.

    //printf("%f, %f, %d\n", fs, tone_freq, channel_count);
    
    // Main loop for tone generation
    buf_out = (float *) pa_buf_out;
    for (i = 0; i < frame_count; i++) {
        for (j = 0; j < channel_count; j++) {
            // Note that we implicitly assume `mix_mat` is an `n x 1` matrix
            buf_out[i*channel_count + j] = (float) (sin(TWOPI * ((float) t) / fs * tone_freq) * ((float) mix_mat->mat[j]));
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
                     unsigned long frame_count,
                     const PaStreamCallbackTimeInfo *time_info,
                     PaStreamCallbackFlags status_flags,
                     void *user_data)
{
    unsigned int i, j, channel_count;
    float *buf_out;
    double fs;
    medussa_dmatrix *mix_mat;
    double tmp;
    white_user_data *wud;
    stream_user_data *sud;
    rk_state *rks; // PRNG variables

    wud = (white_user_data *) user_data;
    sud = (stream_user_data *) wud->parent;

    execute_commands_in_pa_callback( sud->command_queues, execute_stream_user_data_command, sud );

    fs = sud->fs;
    rks = wud->rks;
    mix_mat = sud->is_muted ? sud->mute_mat : sud->mix_mat;
   
    channel_count = sud->out_param->channelCount;
    assert( mix_mat->mat_0 == channel_count ); // matrix must have as many output channels as our stream
    assert( mix_mat->mat_1 == 1 ); // matrix must have 1 source channel: our noise generator.

    // Main loop for tone generation
    buf_out = (float *) pa_buf_out;
    for (i = 0; i < frame_count; i++) {
        for (j = 0; j < channel_count; j++) {
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
            buf_out[i*channel_count + j] = (float) (tmp * ((float) mix_mat->mat[j]));
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
                    unsigned long frame_count,
                    const PaStreamCallbackTimeInfo *time_info,
                    PaStreamCallbackFlags status_flags,
                    void *user_data)
{
    unsigned int i, j, channel_count;
    float *buf_out;
    double fs;
    medussa_dmatrix *mix_mat;
    double tmp;
    pink_user_data *pud;
    stream_user_data *sud;
    pink_noise_t *pn;

    pud = (pink_user_data *) user_data;
    sud = (stream_user_data *) pud->parent;

    execute_commands_in_pa_callback( sud->command_queues, execute_stream_user_data_command, sud );

    fs = sud->fs;
    pn = (pink_noise_t *) pud->pn;

    mix_mat = sud->is_muted ? sud->mute_mat : sud->mix_mat;
    
    channel_count = sud->out_param->channelCount;
    assert( mix_mat->mat_0 == channel_count ); // matrix must have as many output channels as our stream
    assert( mix_mat->mat_1 == 1 ); // matrix must have 1 source channel: our noise generator.

    // Main loop for tone generation
    buf_out = (float *) pa_buf_out;
    for (i = 0; i < frame_count; i++) {
        for (j = 0; j < channel_count; j++) {
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
            buf_out[i*channel_count + j] = (float) (tmp * ((float) mix_mat->mat[j]));
        }
    }

    return paContinue;
}

