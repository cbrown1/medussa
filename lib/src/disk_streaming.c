#include "disk_streaming.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef WIN32
#include <Windows.h>
#include <process.h>
#else 
// posix
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#endif

//#define TRACE( x ) printf x ;
#define TRACE( x )


/*
    DISK STREAMING OVERVIEW

    = I/O Thread =

    There is a single i/o thread that performs all disk i/o operations. 
    When a stream needs to perform an i/o operation (e.g. to read from 
    the sound file) the stream sends a command to the i/o thread to perform 
    the read. Once the read has been performed the i/o thread sends the 
    data back to the stream.

    = I/O Buffers =
    
    Blocks of audio samples are stored in IOBuffers. These buffers are passed 
    around in queues and stored in various lists. The IOBufferList data structure
    is used to maintain linked lists of IOBuffers -- within a single thread, these
    lists are used for LIFO stacks and FIFO queues. IOBuffers are passed 
    between threads using PaUtilRingBuffers.

    Each stream has its own set of IOBuffers. The buffers reference the stream
    and store enough information so that the i/o thread can perform disk i/o
    on the buffer and send it back to the owning stream.

    = File Streams =

    Each file stream is managed by a separate FileStream structure.
    
    Each FileStream maintains it's own pool of IOBuffers. At any time, 
    a FileStream's buffers are in one of three places:
        1. FileStream::free_buffers_lifo -- unused buffers list
        2. FileStream::completed_read_buffers -- buffers that have been read but 
                                                    not yet consumed by the client. kept in sequence
        3. Enqueued with the i/o thread (see below)

    The FileStream always reads ahead and works to keep completed_read_buffers full
    of valid data read from disk. The client consumes data from the head of this list
    by calling file_stream_get_read_buffer_ptr() to get the data, and 
    file_stream_advance_read_ptr() to advance through the data. It is not necessary
    for the client to consume the whole head buffer in one go.
    
    Once a client has consumed a buffer, FileStream moves the buffer to free_buffers_lifo, 
    and from there the buffer is dispatched to read the next block of data from disk.
    This process is described in more detail below.

    When first loading, or after a seek operation, FileStream enters the BUFFERING state
    until all buffers have been read (completed_read_buffers == buffer_count). Once all
    buffers have been read the stream enters the STREAMING state and issues additional 
    read requests when buffers become available.

    FileStream reads buffers in a continuous loop. When it reaches the end of the file
    it begins reading from the start again. This ensures that it's always possible to 
    loop a file smoothly without entering the BUFFERING state.

    = Buffer read life-cycle =

    When a FileStream wants to read a buffer from disk it posts an IO_COMMAND_READ command
    to the i/o thread using enqueue_iocommand_from_pa_callback(). The command contains a 
    pointer to an IOBuffer. When the i/o thread receives the buffer it queues it in the 
    pending_reads_fifo and performs the reads one at a time. It is possible for a stream
    to request that pending reads be canceled using the IO_COMMAND_CANCEL command. This
    allows seeking without waiting for all pending (and now pointless) reads to be completed.

    In more detail, when there is a free IOBuffer available in free_buffers_lifo:

        - FileStream pops an IOBuffer from free_buffers_lifo

        - The buffer is configured with the appropriate file position position_frames 
            for the next read operation in sequence.

        - The buffer is sent from the PA callback to the i/o thread using an IO_COMMAND_READ 
            command passed to enqueue_iocommand_from_pa_callback()

        - In the i/o thread, the i/o thread dequeues the read command, and places the 
            buffer into its pending_reads_fifo queue.

        - The i/o thread works through the pending_reads_fifo queue, reading each buffer
            from disk in turn.

        - When the data has been read from disk, the i/o thread enqueues the buffer
            into the owning FileStream's buffers_from_io_thread queue.

        - Back in the PA callback thread, the FileStream polls its buffers_from_io_thread
            queue from time to time. When the buffer is received in the 
            buffers_from_io_thread queue it removed from the queue and placed on the end 
            of the completed_read_buffers queue.

    = I/O Thread life-cycle =

    The i/o thread is reference counted by acquire_iothread()/release_iothread() pairs,
    which are called when a FileStream is created and destroyed. So long as there is a
    a single FileStream open the i/o thread keeps running.
*/

// ---------------------------------------------------------------------------
// i/o buffer

IOBuffer *allocate_iobuffer( FileStream *file_stream, size_t capacity_frames )
{
    IOBuffer *result = (IOBuffer*)malloc( sizeof(IOBuffer) );
    if( !result )
        return NULL;

    result->data = (double*)malloc( sizeof(double) * capacity_frames * file_stream->channel_count );
    if( !result->data ){
        free( result );
        return NULL;
    }

    result->capacity_frames = capacity_frames;
    result->file_stream = file_stream;
    result->next = NULL;
    result->position_frames = IOBUFFER_INVALID_POSITION;
    result->valid_frame_count = 0;

    return result;
}

void free_iobuffer( IOBuffer *buffer )
{
    free( buffer->data );
    free( buffer );
}


// ---------------------------------------------------------------------------
// i/o buffer list

void IOBufferList_initialize( IOBufferList *list )
{
    list->head = NULL;
    list->tail = NULL;
}

int IOBufferList_is_empty( IOBufferList *list )
{
    return (list->head == NULL);
}

void IOBufferList_push_head( IOBufferList *list, IOBuffer *b )
{
    assert( b->next == NULL );  // must not already be in a list

    b->next = list->head;
    list->head = b;
    if( list->tail == NULL )
        list->tail = list->head;
}

IOBuffer* IOBufferList_pop_head(IOBufferList *list )
{
    IOBuffer *result;

    if( IOBufferList_is_empty(list) )
        return NULL;

    result = list->head;

    list->head = result->next;
    if( list->head == NULL )
        list->tail = NULL;

    result->next = NULL;
    return result;
}

void IOBufferList_push_tail( IOBufferList *list, IOBuffer *b )
{
    if( list->tail ){
        list->tail->next = b;
        list->tail = b;
    }else{
        list->head = list->tail = b;
    }
}

void IOBufferList_prepend_b_to_a( IOBufferList *a, IOBufferList *b )
{
    if( IOBufferList_is_empty(b) )
        return;

    b->tail->next = a->head; // link a->head at the tail of b (also works if a is empty)
    a->head = b->head; // b head becomes a head. (a tail remains a tail)
    b->head = b->tail = NULL; // reset b
}

void IOBufferList_insert_ordered_by_sequence_number( IOBufferList *list, IOBuffer *b )
{
    if( IOBufferList_is_empty(list) ){ // list is empty

        list->head = list->tail = b;

    }else if( b->sequence_number >= list->tail->sequence_number ){ // b should be after tail

        list->tail->next = b;
        list->tail = b;

    }else if( b->sequence_number <= list->head->sequence_number ){ // b should be before head

        b->next = list->head;
        list->head = b;

    }else{ // search for correct position
        IOBuffer *previous = list->head;
        IOBuffer *current = list->head->next;
        while( b->sequence_number >= current->sequence_number ){
            previous = current;
            current = current->next;
        }
        assert( b->sequence_number >= previous->sequence_number && b->sequence_number < current->sequence_number );
        b->next = current;
        previous->next = b;
    }
}

// ---------------------------------------------------------------------------
// file stream

static int round_up_to_next_power_of_2( int x )
{
    x--;
    x |= x >> 1;  // handle  2 bit numbers
    x |= x >> 2;  // handle  4 bit numbers
    x |= x >> 4;  // handle  8 bit numbers
    x |= x >> 8;  // handle 16 bit numbers
    x |= x >> 16; // handle 32 bit numbers
    x++;
     
    return x;
}

static void file_stream_process_buffers_from_io_thread( FileStream *file_stream );

FileStream *allocate_file_stream( SNDFILE *sndfile, const SF_INFO *sfinfo, int buffer_count, int buffer_frame_count )
{
    int i;
    void *ringbuffer_data;
    int ringbuffer_item_count = round_up_to_next_power_of_2( buffer_count );
    FileStream *result = NULL;

    if( sfinfo->frames == 0 )
        return NULL; // don't even try to play a zero length file

    if( acquire_iothread() != IOTHREAD_SUCCESS )
        return NULL;

    result = (FileStream*)malloc(sizeof(FileStream));
    if( !result )
        return NULL;

    memset( result, 0, sizeof(FileStream) );

    ringbuffer_data = malloc( ringbuffer_item_count * sizeof(IOBuffer*) );
    if( !ringbuffer_data ){
        free( result );
        return NULL;
    }

    result->buffer_count = buffer_count;
    result->file_frame_count = sfinfo->frames;
    result->channel_count = sfinfo->channels;

    result->state = FILESTREAM_STATE_IDLE;

    IOBufferList_initialize( &result->free_buffers_lifo );
    result->free_buffers_count = 0;

    TRACE(("ringbuffer_item_count: %d\n", ringbuffer_item_count))
    PaUtil_InitializeRingBuffer( &result->buffers_from_io_thread, sizeof(IOBuffer*), ringbuffer_item_count, ringbuffer_data );

    IOBufferList_initialize( &result->completed_read_buffers );
    result->completed_read_buffer_count = 0;

    result->next_read_sequence_number = 0;
    result->next_completed_read_sequence_number = 0;

    result->next_read_position_frames = 0;
    result->current_position_frames = 0;

    // allocate i/o buffers

    for( i=0; i < buffer_count; ++i ){
        IOBuffer *b = allocate_iobuffer( result, buffer_frame_count );
        if( !b ){
            free_file_stream( result );
            return NULL;
        }
        IOBufferList_push_head( &result->free_buffers_lifo, b );
        ++result->free_buffers_count;
    }

    result->sndfile = sndfile;
    sf_seek(sndfile, 0, SEEK_SET);
    result->sndfile_position_frames = 0;

    return result;
}

static int file_stream_pending_read_count( FileStream *file_stream )
{
    return file_stream->free_buffers_count + file_stream->completed_read_buffer_count;
}

void free_file_stream( FileStream *file_stream )
{
    file_stream_process_buffers_from_io_thread( file_stream );

    // wait until all buffers have been received back from io thread before disposing the file stream
    if( file_stream_pending_read_count(file_stream) < file_stream->buffer_count ){
        IOCommand cmd;
        cmd.action = IO_COMMAND_CANCEL;
        cmd.data.file_stream = file_stream;
        enqueue_iocommand_from_main_thread( &cmd );

        do{
#ifdef WIN32
            Sleep(10);
#else
            usleep(10000);
#endif
            file_stream_process_buffers_from_io_thread( file_stream );
        } while( file_stream_pending_read_count(file_stream) < file_stream->buffer_count );
    }

    while( !IOBufferList_is_empty( &file_stream->completed_read_buffers ) )
        free_iobuffer( IOBufferList_pop_head( &file_stream->completed_read_buffers ) );
    
    while( !IOBufferList_is_empty( &file_stream->free_buffers_lifo ) )
        free_iobuffer( IOBufferList_pop_head( &file_stream->free_buffers_lifo ) );

    free( file_stream->buffers_from_io_thread.buffer );
    free( file_stream );

    release_iothread();
}

void file_stream_issue_read_commands( FileStream *file_stream )
{
    while( file_stream->free_buffers_count > 0 ){
        
        IOBuffer *b = IOBufferList_pop_head( &file_stream->free_buffers_lifo );
        --file_stream->free_buffers_count;

        b->position_frames = file_stream->next_read_position_frames;
        b->sequence_number = file_stream->next_read_sequence_number++;

        file_stream->next_read_position_frames += b->capacity_frames;
        if( file_stream->next_read_position_frames >= file_stream->file_frame_count )
            file_stream->next_read_position_frames = 0; // wrap around and read from start if we reach the end

        {
            IOCommand cmd;
            cmd.action = IO_COMMAND_READ;
            cmd.data.buffer = b;
            enqueue_iocommand_from_pa_callback( &cmd );
        }   
    }
}

void file_stream_seek( FileStream *file_stream, sf_count_t position )
{
    TRACE(("file_stream_seek: %d\n", position))

    if( (file_stream->state == FILESTREAM_STATE_BUFFERING || file_stream->state == FILESTREAM_STATE_STREAMING) 
            && position == file_stream->current_position_frames ){

        TRACE(("file_stream_seek: redundant seek\n"))
        return;
    }
    
    // if there are pending reads, cancel them
    // mark all buffers as free
    // issue new reads

    if( file_stream_pending_read_count(file_stream) > 0 ){
        IOCommand cmd;
        cmd.action = IO_COMMAND_CANCEL;
        cmd.data.file_stream = file_stream;
        enqueue_iocommand_from_pa_callback( &cmd );
    }

    IOBufferList_prepend_b_to_a( &file_stream->free_buffers_lifo, &file_stream->completed_read_buffers );

    file_stream->free_buffers_count += file_stream->completed_read_buffer_count;
    file_stream->completed_read_buffer_count = 0;
    assert( file_stream->free_buffers_count == file_stream->buffer_count );

    file_stream->next_read_position_frames = position;
    file_stream->next_completed_read_sequence_number = file_stream->next_read_sequence_number; // discard canceled and pending buffers prior to the next read
    file_stream->current_position_frames = position;

    file_stream_issue_read_commands( file_stream );

    file_stream->state = FILESTREAM_STATE_BUFFERING;
}

static void file_stream_process_buffers_from_io_thread( FileStream *file_stream )
{
    // iterate all buffers in buffers_from_io_thread
    // if the buffer is valid insert it into completed_read_buffers (see below for definition of validity)
    // otherwise put it on the free list

    IOBuffer *b;

    TRACE(("file_stream_process_buffers_from_io_thread\n"))
    while( PaUtil_ReadRingBuffer( &file_stream->buffers_from_io_thread, &b, 1) ){
        // we expect to receive buffers back from the io thread in the order they were issued
        // based on this we can filter buffers before the last seek based on sequence number

        if( b->sequence_number == file_stream->next_completed_read_sequence_number ){

            TRACE(("file_stream_process_buffers_from_io_thread: OK %p\n", b))

            assert( b->valid_frame_count > 0 );

            IOBufferList_insert_ordered_by_sequence_number( &file_stream->completed_read_buffers, b );
            ++file_stream->completed_read_buffer_count;

            ++file_stream->next_completed_read_sequence_number;

        }else{
            TRACE(("file_stream_process_buffers_from_io_thread: FREE %p\n", b))

            IOBufferList_push_head( &file_stream->free_buffers_lifo, b );
            ++file_stream->free_buffers_count;
        }
    }

    file_stream_issue_read_commands( file_stream );

    if( file_stream->state == FILESTREAM_STATE_BUFFERING && file_stream->completed_read_buffer_count == file_stream->buffer_count )
        file_stream->state = FILESTREAM_STATE_STREAMING;
}

void file_stream_post_buffer_from_iothread( FileStream* file_stream, IOBuffer *buffer )
{
    // assume that the write always succeeds. which is safe so long as the queue has as many slots as a stream has buffers
    PaUtil_WriteRingBuffer( &file_stream->buffers_from_io_thread, &buffer, 1 );
}

sf_count_t file_stream_get_read_buffer_ptr( FileStream *file_stream, double **ptr )
{
    file_stream_process_buffers_from_io_thread( file_stream );

    if( file_stream->state == FILESTREAM_STATE_STREAMING ){
        if( IOBufferList_is_empty( &file_stream->completed_read_buffers ) ){
            file_stream->state = FILESTREAM_STATE_BUFFERING;
            *ptr = NULL;
            TRACE(("file_stream_get_read_buffer_ptr 0\n"))
            return 0;

        }else{
            IOBuffer *b = file_stream->completed_read_buffers.head;
            sf_count_t frame_offset = file_stream->current_position_frames - b->position_frames;
            
            assert( file_stream->current_position_frames >= b->position_frames );

            *ptr = b->data + (frame_offset * file_stream->channel_count);
            TRACE(("file_stream_get_read_buffer_ptr %d\n",  b->valid_frame_count - frame_offset))
            return b->valid_frame_count - frame_offset;
        }
    }else{
        *ptr = NULL;
        TRACE(("file_stream_get_read_buffer_ptr 0\n"))
        return 0;
    }
}

void file_stream_advance_read_ptr( FileStream *file_stream, sf_count_t frame_count )
{
    IOBuffer *b;
    sf_count_t frame_offset;

    file_stream->current_position_frames += frame_count;

    b = file_stream->completed_read_buffers.head;
    frame_offset = file_stream->current_position_frames - b->position_frames;
    assert( frame_offset <=  b->valid_frame_count ); // frame_count should only advance within the current buffer

    if( frame_offset == b->valid_frame_count ){
        IOBufferList_pop_head( &file_stream->completed_read_buffers );
        --file_stream->completed_read_buffer_count;

        IOBufferList_push_head( &file_stream->free_buffers_lifo, b );
        ++file_stream->free_buffers_count;

        file_stream_issue_read_commands( file_stream );
    }

    if( file_stream->current_position_frames >= file_stream->file_frame_count )
        file_stream->current_position_frames = 0;
}

// ---------------------------------------------------------------------------
// i/o thread

#ifdef WIN32

/* use CreateThread for CYGWIN, _beginthreadex for all others */
#if !defined(__CYGWIN__) && !defined(_WIN32_WCE)
#define CREATE_THREAD (HANDLE)_beginthreadex( 0, 0, io_thread_proc, iothread_, 0, &iothread_->thread_id )
#define WIN_THREAD_FUNC static unsigned WINAPI
#define WIN_THREAD_ID unsigned
#else
#define CREATE_THREAD CreateThread( 0, 0, io_thread_proc, iothread_, 0, &iothread_->thread_id )
#define WIN_THREAD_FUNC static DWORD WINAPI
#define WIN_THREAD_ID DWORD
#endif

#endif


typedef struct IOThread{
    volatile int run;

#ifdef WIN32
    HANDLE thread_handle;
    WIN_THREAD_ID thread_id;
    int command_event_inited;
    HANDLE command_event;
#else
    // posix
    pthread_t thread;
    int command_semaphore_inited;
    sem_t command_semaphore; // signaled when there are commands to process
#endif

#define FROM_MAIN_THREAD 0
#define FROM_PA_CALLBACK 1
    PaUtilRingBuffer incoming_commands[ 2 ];

    IOBufferList pending_reads_fifo;
} IOThread;

#define IOTHREAD_COMMAND_QUEUE_COMMAND_COUNT  (1024) /* must be a power of 2 */

static IOThread *iothread_ = NULL; // singleton
static int iothread_refcount_ = 0;

static void iothread_process_commands()
{
    IOCommand cmd;
    int i;

    for( i=0; i < 2; ++i ){
        while( PaUtil_ReadRingBuffer( &iothread_->incoming_commands[i], &cmd, 1) ){
            switch( cmd.action ){
            case IO_COMMAND_READ:
                TRACE(("iothread_process_commands: IO_COMMAND_READ %p\n", cmd.data.buffer))
                IOBufferList_push_tail( &iothread_->pending_reads_fifo, cmd.data.buffer );
                break;
            case IO_COMMAND_CANCEL:
                // walk through pending reads returning all buffers that belong to stream
                {
                    FileStream *stream_to_flush = cmd.data.file_stream;

                    IOBuffer *b = iothread_->pending_reads_fifo.head;
                    IOBuffer **previous_ptr = &iothread_->pending_reads_fifo.head;
                    iothread_->pending_reads_fifo.tail = NULL; // recompute tail setting it to the last non-flushed buffer
                    while( b ){
                        if( b->file_stream == stream_to_flush ){
                            IOBuffer *next = b->next; // capture this before posting the buffer back
                            *previous_ptr = next; // unlink
                            b->next = NULL;
                            b->position_frames = IOBUFFER_INVALID_POSITION;
                            b->valid_frame_count = 0;
                            TRACE(("iothread_process_commands: IO_COMMAND_CANCEL %p\n", b))
                            file_stream_post_buffer_from_iothread( b->file_stream, b );
                            b = next;
                        }else{
                            iothread_->pending_reads_fifo.tail = b;
                            previous_ptr = &b->next;
                            b = b->next;
                        }
                    }
                }

                break;
            }
        }
    }
}

static void iothread_process_pending_io() // returns when there is nothing left to do
{
    iothread_process_commands();

    while( !IOBufferList_is_empty( &iothread_->pending_reads_fifo ) ){

        // perform a read and return the result to the pa callback

        IOBuffer *b = IOBufferList_pop_head( &iothread_->pending_reads_fifo );
        
        if( b->position_frames != b->file_stream->sndfile_position_frames ){
            sf_seek(b->file_stream->sndfile, b->position_frames, SEEK_SET);
            b->file_stream->sndfile_position_frames = b->position_frames;
        }

        TRACE(("sf_readf_double %d\n", b->capacity_frames))
        b->valid_frame_count = sf_readf_double( b->file_stream->sndfile, b->data, b->capacity_frames );
        b->file_stream->sndfile_position_frames += b->valid_frame_count;

        file_stream_post_buffer_from_iothread( b->file_stream, b );

        iothread_process_commands();
    }
}

#ifdef WIN32

WIN_THREAD_FUNC io_thread_proc( void *pArg )
{
    while( iothread_->run ){
        iothread_process_pending_io();
        WaitForSingleObject( iothread_->command_event, 1000 );
    }

    iothread_process_pending_io();

    return 0;
}

#else
    // posix

static void *io_thread_func( void *param )
{
    while( iothread_->run ){
        iothread_process_pending_io();
        sem_wait( &iothread_->command_semaphore );
    }

    iothread_process_pending_io();

    return 0;
}

#endif

static int create_iothread() // returns 0 on success
{
    void *ringbuffer_data = 0;
    int i;

    assert( iothread_ == NULL );

    iothread_ = (IOThread*)malloc( sizeof(IOThread) );
    if( !iothread_ )
        goto fail;

    memset( iothread_, 0, sizeof(IOThread) );

    for( i=0; i < 2; ++i )
        iothread_->incoming_commands[i].buffer = 0;

    for( i=0; i < 2; ++i ){
        ringbuffer_data = malloc( IOTHREAD_COMMAND_QUEUE_COMMAND_COUNT * sizeof(IOCommand) );
        if( !ringbuffer_data )
            goto fail;

        PaUtil_InitializeRingBuffer( &iothread_->incoming_commands[i], sizeof(IOCommand), IOTHREAD_COMMAND_QUEUE_COMMAND_COUNT, ringbuffer_data );
    }

    iothread_->run = 1;

#ifdef WIN32
    iothread_->command_event_inited = 0;

    iothread_->command_event = CreateEvent( NULL, /* bManualReset= */ FALSE, /* bInitialState= */ TRUE, NULL );
    if( iothread_->command_event == NULL )
        goto fail;

    iothread_->command_event_inited = 1;

    iothread_->thread_handle = CREATE_THREAD;
    if( iothread_->thread_handle == NULL )
        goto fail;

    SetThreadPriority( iothread_->thread_handle, THREAD_PRIORITY_ABOVE_NORMAL );  // prioritize disk io above normal but below real-time audio

#else
    // posix
    {
        int pthread_result;
        pthread_attr_t attr;

        iothread_->command_semaphore_inited = 0;

        if( sem_init( &iothread_->command_semaphore, 0, 0 ) != 0 )
            goto fail;

        iothread_->command_semaphore_inited = 1;

        pthread_attr_init( &attr );
        pthread_attr_setscope( &attr, PTHREAD_SCOPE_SYSTEM );
        pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );

        pthread_result = pthread_create( &iothread_->thread, &attr, io_thread_func, iothread_ );
        pthread_attr_destroy( &attr );
        if( pthread_result != 0 )
            goto fail;
    }
#endif

    return IOTHREAD_SUCCESS;

fail:
    TRACE(("create_iothread: FAIL"))

    if( iothread_ ){
        
        for( i=0; i < 2; ++i ){
            if( iothread_->incoming_commands[i].buffer )
                free( iothread_->incoming_commands[i].buffer );
        }

#ifdef WIN32

        if( iothread_->command_event_inited )
            CloseHandle( iothread_->command_event );

#else
        // posix

        if( iothread_->command_semaphore_inited )
            sem_destroy( &iothread_->command_semaphore );
#endif

        free( iothread_ );
        iothread_ = NULL;
    }

    return IOTHREAD_FAIL;
}

static void destroy_iothread()
{
    int i;

    assert( IOBufferList_is_empty(&iothread_->pending_reads_fifo) );

    iothread_->run = 0;

#ifdef WIN32
    SetEvent( iothread_->command_event );

    WaitForSingleObject( iothread_->thread_handle, 1000 );
    CloseHandle( iothread_->thread_handle );

    CloseHandle( iothread_->command_event );
#else
    // posix
    void *thread_result;
    sem_post( &iothread_->command_semaphore );
    int pthread_result = pthread_join( iothread_->thread, &thread_result );
    assert( pthread_result == 0 );

    sem_destroy( &iothread_->command_semaphore );
#endif

    for( i=0; i < 2; ++i ){
        if( iothread_->incoming_commands[i].buffer )
            free( iothread_->incoming_commands[i].buffer );
    }
    free( iothread_ );
    iothread_ = NULL;
}

int acquire_iothread()
{
    int result = IOTHREAD_SUCCESS;

    if( iothread_refcount_ == 0 ){ // thread not running

        result = create_iothread();
        if( result == IOTHREAD_SUCCESS )
            ++iothread_refcount_; // only inc refcount if the thread was created

    }else{
        ++iothread_refcount_;
    }

    return result;
}

void release_iothread()
{
    assert( iothread_refcount_ > 0 ); // no refs. released too many times?

    if( --iothread_refcount_ == 0 )
        destroy_iothread();
}

void enqueue_iocommand_from_pa_callback( const IOCommand *command )
{
    assert( iothread_ != NULL ); // didn't call acquire_iothread?

    // this will always succeed if there are more command slots than MAX_ACTIVE_STREAMS * (BUFFERS_PER_STREAM + 1) 
    // i.e. each stream can have BUFFERS_PER_STREAM and a cancel request pending
    PaUtil_WriteRingBuffer( &iothread_->incoming_commands[FROM_PA_CALLBACK], command, 1 );

#ifdef WIN32
    SetEvent( iothread_->command_event );
#else
    sem_post( &iothread_->command_semaphore );
#endif
}

void enqueue_iocommand_from_main_thread( const IOCommand *command )
{
    assert( iothread_ != NULL ); // didn't call acquire_iothread?

    // this will always succeed if there are more command slots than MAX_ACTIVE_STREAMS * (BUFFERS_PER_STREAM + 1) 
    // i.e. each stream can have BUFFERS_PER_STREAM and a cancel request pending
    PaUtil_WriteRingBuffer( &iothread_->incoming_commands[FROM_MAIN_THREAD], command, 1 );

#ifdef WIN32
    SetEvent( iothread_->command_event );
#else
    sem_post( &iothread_->command_semaphore );
#endif
}

// ---------------------------------------------------------------------------