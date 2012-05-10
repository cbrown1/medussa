#include "medussa_types.h"

#include <stdlib.h>
#include <string.h>


/* -------------------------------------------------------------------- */

/* Allocate a medussa_dmatrix of dimension mat_0, mat_1 and initialize with data from mat.
   If mat is NULL, the matrix contains all zeros.
   If a memory allocation failure occurs the function returns NULL.
*/
medussa_dmatrix* alloc_medussa_dmatrix( int mat_0, int mat_1, double *mat )
{
    int dataLengthBytes = sizeof(double) * mat_0 * mat_1;

    medussa_dmatrix *result = (medussa_dmatrix*)malloc( sizeof(medussa_dmatrix) );
    if( !result )
        return NULL;

    result->mat_0 = mat_0;
    result->mat_1 = mat_1;
    
    result->mat = (double*)malloc( dataLengthBytes );
    if( !result->mat ){
        free( result );
        return NULL;
    }

    if( mat )
        memcpy( result->mat, mat, dataLengthBytes );
    else
        memset( result->mat, 0, dataLengthBytes );

    return result;
}

/* Free a medussa_dmatrix previously allocated with alloc_medussa_dmatrix */
void free_medussa_dmatrix( medussa_dmatrix *mat )
{
    if( !mat ) return;
    free( mat->mat );
    free( mat );
}

/* -------------------------------------------------------------------- */

#define STREAM_COMMAND_QUEUE_COMMAND_COUNT  (1024) /* must be a power of 2 */

stream_command_queues* alloc_stream_command_queues()
{
    void *data0, *data1;
    stream_command_queues *result = (stream_command_queues*)malloc( sizeof(stream_command_queues) );
    if( !result )
        return NULL;

    data0 = malloc( STREAM_COMMAND_QUEUE_COMMAND_COUNT * sizeof(stream_command) );
    if( !data0 ){
        free( result );
        return NULL;
    }

    data1 = malloc( STREAM_COMMAND_QUEUE_COMMAND_COUNT * sizeof(stream_command) );
    if( !data1 ){
        free( result );
        free( data0 );
        return NULL;
    }

    PaUtil_InitializeRingBuffer( &result->from_python_to_pa_callback, sizeof(stream_command), STREAM_COMMAND_QUEUE_COMMAND_COUNT, data1 );
    PaUtil_InitializeRingBuffer( &result->from_pa_callback_to_python, sizeof(stream_command), STREAM_COMMAND_QUEUE_COMMAND_COUNT, data0 );
    
    return result;
}

void free_stream_command_queues( stream_command_queues* qs )
{
    if( !qs ) return;

    free( qs->from_python_to_pa_callback.buffer );
    free( qs->from_pa_callback_to_python.buffer );
    free( qs );
}

int post_command_to_pa_callback( stream_command_queues* qs, const stream_command *cmd )
{
    return PaUtil_WriteRingBuffer( &qs->from_python_to_pa_callback, cmd, 1 );
}

/* Dequeue any commands received from pa_callback in the python thread,
execute them directly. (these are usually deallocation requests).
*/
void process_results_from_pa_callback( stream_command_queues* qs )
{
    stream_command cmd;
    while( PaUtil_ReadRingBuffer( &qs->from_pa_callback_to_python, &cmd, 1) ){
        switch( cmd.command ){
            case STREAM_COMMAND_FREE_MATRICES:
                free_medussa_dmatrix( (medussa_dmatrix*)cmd.data_ptr0 );
                free_medussa_dmatrix( (medussa_dmatrix*)cmd.data_ptr1 );
                break;
        }
    }
}

/* Dequeue any commands received from python int the pa_callback,
execute them using executeCommandFunction
*/
void execute_commands_in_pa_callback( stream_command_queues* qs, execute_command_function_ptr executeCommandFunction, void *data )
{
    stream_command cmd;
    while( PaUtil_ReadRingBuffer( &qs->from_python_to_pa_callback, &cmd, 1) ){
        (*executeCommandFunction)( &qs->from_pa_callback_to_python, &cmd, data );
    }
}

/* -------------------------------------------------------------------- */

