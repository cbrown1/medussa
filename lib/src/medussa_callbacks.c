/*
# Copyright (c) 2010-2012 Christopher Brown
#
# This file is part of Medussa.
#
# Medussa is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Medussa is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Medussa.  If not, see <http://www.gnu.org/licenses/>.
#
# Comments and/or additions are welcome. Send e-mail to: cbrown1@pitt.edu.
#
*/
#define _USE_MATH_DEFINES
#include "medussa_callbacks.h"
#include "medussa_matrix.h"
#include "log.h"

#include <math.h>

#define TWOPI 6.2831853071795862


void execute_stream_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data )
{
    stream_user_data *sud = (stream_user_data *) data;
    stream_command resultCommand;

    switch( command->command ){
    case STREAM_COMMAND_SET_MATRICES:
        {
            resultCommand.command = STREAM_COMMAND_FREE_MATRICES;
            resultCommand.data_ptr0 = 0;
            resultCommand.data_ptr1 = 0;

            sud->mix_mat_fade_countdown_frames = command->data_uint;
            sud->mix_mat_fade_total_frames = sud->mix_mat_fade_countdown_frames;
            debug("execute_stream_user_data_command SET_MATRICES frames=%u",
                    sud->mix_mat_fade_total_frames);
            if( sud->mix_mat_fade_countdown_frames == 0 ){

                /* no fade. send back the old mix and target matrices back and install a new mix matrix */
                resultCommand.data_ptr0 = sud->mix_mat;
                sud->mix_mat = (medussa_dmatrix*)command->data_ptr0;

                resultCommand.data_ptr1 = sud->target_mix_mat;
                sud->target_mix_mat = 0;

            }else{
                /* compute fade increment matrix into fade_mat. put new matrix into target_mix_mat */
                resultCommand.data_ptr0 = sud->target_mix_mat;
                sud->target_mix_mat = (medussa_dmatrix*)command->data_ptr0;
                /* fade_inc_mat = (target_mix_mat - mix_mat) * (1. / sud->mix_mat_fade_countdown_frames); */

                dmatrix_subtract( sud->target_mix_mat->mat, sud->target_mix_mat->mat_0, sud->target_mix_mat->mat_1,
                                    sud->mix_mat->mat, sud->mix_mat->mat_0, sud->mix_mat->mat_1,
                                    sud->fade_inc_mat->mat, sud->fade_inc_mat->mat_0, sud->fade_inc_mat->mat_1 );

                if( !(sud->flags & STREAM_FLAG_COSINE_FADE) ){
                    debug("using linear mix_mat fade");
                    dmatrix_scale( sud->fade_inc_mat->mat, sud->fade_inc_mat->mat_0, sud->fade_inc_mat->mat_1,
                                        (1. / sud->mix_mat_fade_countdown_frames),
                                        sud->fade_inc_mat->mat, sud->fade_inc_mat->mat_0, sud->fade_inc_mat->mat_1 );
                } else {
                    debug("using cosine mix_mat fade");
                }
            }

            /* post old matrices back to main python thread to be freed */
            PaUtil_WriteRingBuffer(resultQueue, &resultCommand, 1 );
        }
        break;

    case STREAM_COMMAND_SET_IS_MUTED:
        sud->is_muted = command->data_uint;
        debug("execute_stream_user_data_command SET_IS_MUTED=%d", sud->is_muted);
        break;
    }
}

static double half_hann(unsigned index, unsigned total) {
    return 0.5 * (1. - cos(M_PI * index / total));
}

void increment_mix_mat_fade( stream_user_data *sud )
{
    unsigned index;
    int cosine;
    if( sud->mix_mat_fade_countdown_frames == 0 )
        return;

    index = sud->mix_mat_fade_total_frames - sud->mix_mat_fade_countdown_frames;
    cosine = (sud->flags & STREAM_FLAG_COSINE_FADE);
    // debug("increment_mix_mat_fade index=%u total=%u cosine=%d",
    //         index, sud->mix_mat_fade_total_frames, cosine);

    if( !cosine ){
        // mix_mat += fade_inc_mat
        dmatrix_add( sud->mix_mat->mat, sud->mix_mat->mat_0, sud->mix_mat->mat_1,
            sud->fade_inc_mat->mat, sud->fade_inc_mat->mat_0, sud->fade_inc_mat->mat_1,
            sud->mix_mat->mat, sud->mix_mat->mat_0, sud->mix_mat->mat_1 );
    } else {
        // XXX we never run for the case when index == total, so need to use index + 1 to
        // avoid jumping over last cosine value when swapping target mat below
        // mix_mat = target_mat - (1 - half_hann(index + 1, total)) * fade_mat
        double y = half_hann(index + 1, sud->mix_mat_fade_total_frames);
        dmatrix_scale( sud->fade_inc_mat->mat, sud->fade_inc_mat->mat_0, sud->fade_inc_mat->mat_1,
            1. - y,
            sud->mix_mat->mat, sud->mix_mat->mat_0, sud->mix_mat->mat_1 );
        dmatrix_subtract( sud->target_mix_mat->mat, sud->target_mix_mat->mat_0, sud->target_mix_mat->mat_1,
            sud->mix_mat->mat, sud->mix_mat->mat_0, sud->mix_mat->mat_1,
            sud->mix_mat->mat, sud->mix_mat->mat_0, sud->mix_mat->mat_1 );
    }
    --sud->mix_mat_fade_countdown_frames;
    if( sud->mix_mat_fade_countdown_frames == 0 ){
        // Swap mix_mat and target_mix_mat.
        // old mix_mat (now stored in target_mix_mat) will be freed next time STREAM_COMMAND_SET_MATRICES executes
        medussa_dmatrix *temp_mat = sud->mix_mat;
        sud->mix_mat = sud->target_mix_mat;
        sud->target_mix_mat = temp_mat;
    }
}


void execute_finite_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data )
{
    finite_user_data *fud = (finite_user_data *) data;

    switch( command->command ){
    case FINITE_STREAM_COMMAND_SET_CURSOR:

        fud->cursor = command->data_uint;
        /* note that we only seek the stream in the pa callback because the stream can usually only communicate with io thread from the callback */
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
    unsigned int array_index;
    int loop;        // Boolean
    float *buf_out;  // Points to `pa_buf_out`
    double *tmp_buf;
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

    // Determine `array_channel_count`, the number of channels, from `arr`
    array_channel_count = (unsigned int) aud->ndarr_1;

    assert( mix_mat->mat_0 == stream_channel_count ); // matrix must have as many output channels as our stream
    assert( mix_mat->mat_1 == array_channel_count ); // matrix must have same number of source channels as the file

    // Point `arr_frames` to C array of `arr`, move cursor appropriately
    arr = aud->ndarr;
    array_index = fud->cursor;

    // Copy each frame from of `arr` to the output buffer, multiplying by
    // the mixing matrix each time.
    tmp_buf = fud->temp_mat->mat;
    buf_out = (float *) pa_buf_out;
    for (i = 0; i < frame_count; i++, array_index++) {

        if( array_index >= (unsigned)aud->ndarr_0 ){
            if( loop )
                array_index = 0;
            else
                break;
        }
        
        dmatrix_mult(mix_mat->mat, mix_mat->mat_0, mix_mat->mat_1,
                     arr + array_index*array_channel_count,
                     array_channel_count, 1,
                     tmp_buf,
                     stream_channel_count, 1);

        for (j = 0; j < stream_channel_count; j++) {
            buf_out[i*stream_channel_count + j] = (float) tmp_buf[j];
        }

        increment_mix_mat_fade( sud );
    }

    // if we're at the end of the source array write silence into the remainder of the output buffer
    for (; i < frame_count; i++) {
         for (j = 0; j < stream_channel_count; j++) {
                buf_out[i*stream_channel_count + j] = 0.0f;
         }
    }

    // Move `self.cursor`
    fud->cursor = array_index; // Assume ATOMIC STORE

    if( fud->cursor >= (unsigned int)aud->ndarr_0 && !loop ){

        // NOTE: if the stream has completed we don't reset the cursor to zero.
        return paComplete;
    }else{

        return paContinue;
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
    int i, j;
    sf_count_t n;
    int loop;
    int stream_channel_count; // Number of stream output channels
    int file_channel_count; // Samples per frame for input file
    double *read_buf;
    sf_count_t read_buffer_frame_count;
    sf_count_t frames_to_go;
    int output_frame_index;
    double *tmp_buf;
    char *finpath;
    SF_INFO *finfo;
    medussa_dmatrix *mix_mat;
    sndfile_user_data *sfud;
    finite_user_data  *fud;
    stream_user_data  *sud;
    int result = paContinue;
    
    sfud = (sndfile_user_data *) user_data;
    fud  = (finite_user_data *)  sfud->parent;
    sud = (stream_user_data *)  fud->parent;

    execute_commands_in_pa_callback( sud->command_queues, execute_sndfile_read_user_data_command, sfud );

    // Begin attribute acquisition
    finfo = sfud->finfo;
    stream_channel_count = sud->out_param->channelCount;
    mix_mat = sud->is_muted ? sud->mute_mat : sud->mix_mat;
    loop = fud->loop;
    finpath = sfud->finpath;
    fin = sfud->fin;
    file_channel_count = finfo->channels;
    // End attribute acquisition

    assert( mix_mat->mat_0 == stream_channel_count ); // matrix must have as many output channels as our stream
    assert( mix_mat->mat_1 == file_channel_count ); // matrix must have same number of source channels as the file

    tmp_buf = fud->temp_mat->mat;
    buf_out = (float *) pa_buf_out;
    output_frame_index = 0;    
    frames_to_go = frame_count;
 
    file_stream_seek( sfud->file_stream, fud->cursor );

    // pull one or more buffer regions from the file stream and matrix it to out_buf
    while( frames_to_go > 0 ){
        read_buffer_frame_count = file_stream_get_read_buffer_ptr( sfud->file_stream, &read_buf );
        if( read_buffer_frame_count == 0 )
            break; // no data available, we write silence below

        n = (frames_to_go<read_buffer_frame_count) ? frames_to_go : read_buffer_frame_count;
        for ( i = 0; i < n; i++, output_frame_index++ ) {
            dmatrix_mult(mix_mat->mat, mix_mat->mat_0, mix_mat->mat_1,
                (read_buf+i*file_channel_count), file_channel_count, 1,
                tmp_buf, stream_channel_count, 1);
            for (j = 0; j < stream_channel_count; j++) {
                buf_out[output_frame_index*stream_channel_count + j] = (float) tmp_buf[j];
            }

            increment_mix_mat_fade( sud );
        }
        frames_to_go -= n;

        // Move `self.cursor`
        fud->cursor = (fud->cursor + (unsigned int)n);  // Assume ATOMIC STORE

        file_stream_advance_read_ptr( sfud->file_stream, n );
        if( sfud->file_stream->current_position_frames == 0 ){ // file stream has reached end and wrapped position to start [FIXME this is a bit brittle]
            if( loop ){
                fud->cursor = 0; // Assume ATOMIC STORE
                result = paContinue;
            }else{
                result = paComplete;
            }
        }
    }

    // write silence into the remainder of the output buffer if we didn't fill it all
    for(; output_frame_index < (int)frame_count; ++output_frame_index ) {
        for (j = 0; j < stream_channel_count; j++) {
            buf_out[output_frame_index*stream_channel_count + j] = 0.0f;
        }
    }

    return result;
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

        increment_mix_mat_fade( sud );
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

        increment_mix_mat_fade( sud );
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

        increment_mix_mat_fade( sud );
    }

    return paContinue;
}
