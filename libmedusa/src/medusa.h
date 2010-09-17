#include <Python.h>
#include <numpy/arrayobject.h>

#include <portaudio.h>
#include <sndfile.h>


#define NDARRAY_STREAM 0
#define SNDFILE_STREAM 1
#define TONE_STREAM 2


// Pointer and metadata for an ndarray object containing floating point samples
typedef struct ContigArrayHandle {
    PyObject *x;      // Refers to the (assumed-contiguous) ndarray
    int chan_i;       // Iterates along dimension 0, must be < PyArray_DIM(x, 0)
    int samp_i;       // Iterates along dimension 1, must be < PyArray_DIM(x, 1)
    double samp_freq; // Sampling frequency in Hertz
    double scale;     // Scaling factor for each sample, should be in the interval [0, 1]
    int loop;         // Boolean to determine whether or not to loop array playback
} ContigArrayHandle;


typedef struct SndfileData {
    void *fin;          // Will be cast as (SNDFILE *) to an input file
    void *fout;         // Will be cast as (SNDFILE *) to an output file
    SF_INFO *fin_info;  // Input file's info struct
    SF_INFO *fout_info; // Output file's info struct
    double scale;       // Scaling factor for each sample, should be in the interval [0, 1]
    int loop;           // Boolean to determine whether or not to loop array playback
    unsigned int time;  // Where we are pointing in the file in units of frames
} SndfileData;


// Data for pure sinusoidal tone, in frames of width `channels`, with a channel orientation
typedef struct ToneData {
    unsigned int t;   // Global `time` variable
    int channels;     // Number of channels total
    int chan_out;     // Output channel
    double tone_freq; // Frequency of the tone that will be generated
    double samp_freq; // Sampling frequency in Hertz
    double scale;     // Scaling factor for each sample, should be in the interval [0, 1]
} ToneData;


// Data for pink noise
typedef struct PinkData {
    double samp_freq;
    double scale;
} PinkData;


int callback_ndarray (const void *,
                      void *,
                      unsigned long,
                      const PaStreamCallbackTimeInfo *,
                      PaStreamCallbackFlags,
                      void *);

int callback_sndfile_read (const void *,
                          void *,
                          unsigned long,
                          const PaStreamCallbackTimeInfo *,
                          PaStreamCallbackFlags,
                          void *);

int callback_tone (const void *,
                   void *,
                   unsigned long,
                   const PaStreamCallbackTimeInfo *,
                   PaStreamCallbackFlags,
                   void *);

int callback_pink (const void *,
                   void *,
                   unsigned long,
                   const PaStreamCallbackTimeInfo *,
                   PaStreamCallbackFlags,
                   void *);


PaStream * open_stream (PaStream *,
                        PaStreamParameters *,
                        PaStreamParameters *,
                        PyObject *,
                        void *,
                        int);

void start_streams   (PaStream **, int);