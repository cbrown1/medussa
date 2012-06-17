#include <Python.h>
#include <numpy/arrayobject.h>

#include <portaudio.h>
#include "pa_ringbuffer.h"
#include <sndfile.h>
#include "randomkit.h"
#include "pink.h"
#include "disk_streaming.h"

/* -------------------------------------------------------------------- */

struct medussa_dmatrix{
    double *mat;
    int mat_0;
    int mat_1;
};
typedef struct medussa_dmatrix medussa_dmatrix;

medussa_dmatrix* alloc_medussa_dmatrix( int mat_0, int mat_1, double *mat ); // pass null data to init the matrix
void free_medussa_dmatrix( medussa_dmatrix *mat );

/* -------------------------------------------------------------------- */

typedef enum {
    STREAM_COMMAND_SET_MATRICES,    /* mix_mat > data_ptr0, mute_mat > data_ptr1 */
    STREAM_COMMAND_FREE_MATRICES,   /* mix_mat > data_ptr0, mute_mat > data_ptr1 */
    STREAM_COMMAND_SET_IS_MUTED,    /* is_muted > data_uint */

    FINITE_STREAM_COMMAND_SET_CURSOR /* cursor > data_uint */
};


/* generic command structure, used for all commands */
struct stream_command{
    int command;
    void *data_ptr0;
    void *data_ptr1;
    double data_double;
    unsigned int data_uint;
};
typedef struct stream_command stream_command;


/* bidirectional command queue pair for sending commands to the 
    callback (from python), and receiving commands from the callback. 
*/
struct stream_command_queues{
    PaUtilRingBuffer from_python_to_pa_callback;
    PaUtilRingBuffer from_pa_callback_to_python;
};
typedef struct stream_command_queues stream_command_queues;

stream_command_queues* alloc_stream_command_queues();
void free_stream_command_queues( stream_command_queues* qs );

/* copy cmd into queue that sends to callback. returns 1 on success, 0 on fail */
int post_command_to_pa_callback( stream_command_queues* qs, const stream_command *cmd );

/* execute result commands such as freeing old data. (call in the python thread)*/
void process_results_from_pa_callback( stream_command_queues* qs );

/* command exec function type. there is a separate one of these for each pa callback. often they are chained to provide behavioral inheritance. */
typedef void (*execute_command_function_ptr)( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data );

/* loop through received commands,  call f on each command to execute it. (call in the pa callback) */
void execute_commands_in_pa_callback( stream_command_queues* qs, execute_command_function_ptr f, void *data );


/*
    usage: 

    call alloc_stream_command_queues() when creating the stream user data
   
    to send a command:

    first call process_commands_from_pa_callback() just to pump the queue and free any previous data.
    populate a local temporary command structure
    call send_command_to_pa_callback.
    if the stream isn't running:
        call the most derived classes execute_stream_user_data_commands method to process the commands immediately.

    freeing:

    when destroying a stream, first call most derived classes execute_stream_user_data_commands to process any pending commands, 
    then process_results_from_pa_callback to ensure that all results are freed
    then call free_stream_command_queues



    make sure python side uses a sleep-retry mechanism if the commandqueue is full
*/


/* -------------------------------------------------------------------- */

struct stream_user_data {
    void *parent;

    PyObject *device;
    
    PaStream *stream;
    PaStreamParameters *in_param;
    PaStreamParameters *out_param;
    double fs;

    stream_command_queues *command_queues;

    int is_muted;
    medussa_dmatrix *mix_mat;
    medussa_dmatrix *mute_mat;

    int pa_fpb;
};
typedef struct stream_user_data stream_user_data;

void execute_stream_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data );


struct finite_user_data {
    void *parent;

    int loop;
    /* While stream is running <cursor> is only written by PA callback and
       hence used for single-writer-single-reader atomic communication: Written by callback, read by Python. */
    volatile unsigned int cursor;
    unsigned int frames;
    double duration;
};
typedef struct finite_user_data finite_user_data;

void execute_finite_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data );


struct array_user_data {
    void *parent;
    PyObject *self;

    double *ndarr;
    int ndarr_0;
    int ndarr_1;
};
typedef struct array_user_data array_user_data;

void execute_array_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data );


struct sndfile_user_data {
    void *parent;
    PyObject *self;

    SNDFILE *fin;
    char *finpath;
    SF_INFO *finfo;

    FileStream *file_stream;
};
typedef struct sndfile_user_data sndfile_user_data;

void execute_sndfile_read_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data );


struct tone_user_data {
    void *parent;
    PyObject *self;

    unsigned int t;
    double tone_freq;
};
typedef struct tone_user_data tone_user_data;

void execute_tone_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data );

struct white_user_data {
    void *parent;
    PyObject *self;

    rk_state *rks;
};
typedef struct white_user_data white_user_data;

void execute_white_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data );

struct pink_user_data {
    void *parent;
    PyObject *self;

    pink_noise_t *pn;
};
typedef struct pink_user_data pink_user_data;

void execute_pink_user_data_command( PaUtilRingBuffer *resultQueue, const stream_command *command, void *data );