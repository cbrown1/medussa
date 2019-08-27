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

#ifndef INCLUDED_DISKSTREAMING_H
#define INCLUDED_DISKSTREAMING_H

#include <stddef.h>
#include <sndfile.h>
#include "pa_ringbuffer.h"

// ---------------------------------------------------------------------------
// i/o buffer is the unit of file i/o.
// the buffer data is read in the i/o thread and used in the pa callback
// ***
// Threading: accessed from one thread at a time. Passed between threads using PaUtilRingBuffer

#define IOBUFFER_INVALID_POSITION (-1)

typedef struct IOBuffer{
    struct IOBuffer *next; // linked list link. the buffer will only appear in one list at a time. NULL when not in a list
    struct FileStream *file_stream;
    sf_count_t position_frames;
    size_t capacity_frames;
    sf_count_t valid_frame_count;
    int sequence_number;
    double *data; // to make things easy this should be dimensioned as a multiple of pabuffer size * file channels
} IOBuffer;

IOBuffer *allocate_iobuffer( struct FileStream *file_stream, size_t capacity_frames );
void free_iobuffer( IOBuffer *buffer );


// ---------------------------------------------------------------------------
// singly linked list of IOBuffers used as a LIFO or FIFO stack or list.
// ***
// Threading: accessed from one thread at a time.

typedef struct IOBufferList{ 
    IOBuffer *head;
    IOBuffer *tail;	// support push_tail and optimize insert_ordered at end
} IOBufferList;

void IOBufferList_initialize( IOBufferList *list );
int IOBufferList_is_empty( IOBufferList *list );
void IOBufferList_push_head( IOBufferList *list, IOBuffer *b );
IOBuffer* IOBufferList_pop_head(IOBufferList *list );
void IOBufferList_push_tail( IOBufferList *list, IOBuffer *b );

void IOBufferList_prepend_b_to_a( IOBufferList *a, IOBufferList *b );

void IOBufferList_insert_ordered_by_sequence_number( IOBufferList *list, IOBuffer *b );


// ---------------------------------------------------------------------------
// file stream handles the state of asynchronous i/o for one open sound file.
// read requests are sent to the i/o thread. the resulting
// buffers are returned asynchronously in buffers_from_io_thread
// each file stream has a fixed number of i/o buffers that it recycles
// i/o buffers for a stream all have the same capacity
// file stream always streams in a continuous loop so that we can seamlessly
// loop from end to start without the buffering pause that would happen if
// we had to seek after we'd reached the end.
// ***
// Threading: data as marked below.

#define FILESTREAM_STATE_IDLE       (0)
#define FILESTREAM_STATE_BUFFERING  (1)
#define FILESTREAM_STATE_STREAMING  (2)
#define FILESTREAM_STATE_ERROR      (3)

typedef struct FileStream{
    // ***
    // Threading: read-only after creation ----------------------------------------------
    int buffer_count;
    sf_count_t file_frame_count;
    size_t channel_count;

    // ***
    // Threading: accessed in PA callback only ------------------------------------------
    int state;

    IOBufferList free_buffers_lifo;			    // LIFO stack of free (unused) buffers
    int free_buffers_count;                     // number of buffers in free_buffers_lifo

    PaUtilRingBuffer buffers_from_io_thread;    // queue to receive IOBuffer * from io thread
    IOBufferList completed_read_buffers;		// ordered list of read buffers. ordered by file position
    int completed_read_buffer_count;            // number of buffers in completed_read_buffers

    int next_read_sequence_number;
    int next_completed_read_sequence_number;

    sf_count_t next_read_position_frames;
    sf_count_t current_position_frames;

    // ***
    // Threading: accessed in i/o thread only --------------------------------------------
    SNDFILE *sndfile;
    sf_count_t sndfile_position_frames; // cache file position to make seeking easier
} FileStream;

// ***
// Threading: allocate and free from main (Python interpreter) thread
FileStream *allocate_file_stream( SNDFILE *sndfile, const SF_INFO *sfinfo, int buffer_count, int buffer_frame_count );
void free_file_stream( FileStream *file_stream );

// ***
// Threading: seek, get_read_buffer_ptr, advance_read_ptr from PA callback only.
void file_stream_seek( FileStream *file_stream, sf_count_t position );

// get the next buffer segment. returns the length of the segment.
sf_count_t file_stream_get_read_buffer_ptr( FileStream *file_stream, double **ptr );
void file_stream_advance_read_ptr( FileStream *file_stream, sf_count_t frame_count );

// ***
// Threading: post_buffer_from_iothread from i/o thread only
void file_stream_post_buffer_from_iothread( FileStream* file_stream, IOBuffer *buffer );

// ---------------------------------------------------------------------------
// i/o commands
// used to request i/o from the i/o thread

#define IO_COMMAND_READ		(0)	// read the specified buffer
#define IO_COMMAND_CANCEL	(1)	// return any pending reads for the specified file_stream

typedef struct IOCommand{
    int action; // IO_COMMAND_READ, IO_COMMAND_FLUSH

    union data{
        IOBuffer *buffer;
        FileStream *file_stream;
    }data;   
} IOCommand;


// ---------------------------------------------------------------------------
// i/o thread
// the i/o thread performs IOCommands enqueued with enqueue_iocommand_from_pa_callback
// clients must call acquire_iothread from the python thread before enqueued 
// commands and release_iothread once no more commands will be queued.

#define IOTHREAD_SUCCESS    (0)
#define IOTHREAD_FAIL       (1)

// i/o thread is reference counted. each stream should acquire it before use and release it afterwards
// ***
// Threading: acquire and release from main (Python) thread. Usually this is handled by the streams.
int acquire_iothread(void); // returns IOTHREAD_SUCCESS on success
void release_iothread(void);

// ***
// Threading: enqueue an i/o command from the PA callback thread
void enqueue_iocommand_from_pa_callback( const IOCommand *command ); // Copies command to i/o thread command queue

// ***
// Threading: enqueue an i/o command from the main (Python interpreter) thread
void enqueue_iocommand_from_main_thread( const IOCommand *command );

#endif /* INCLUDED_DISKSTREAMING_H */
