#include <stdlib.h>
#include <stdio.h>

#include <portaudio.h>

#include "callback.h"

#define ERROR_CHECK {                                                              \
                        if (err != paNoError) {                                    \
                            printf("PortAudio error: %s\n", Pa_GetErrorText(err)); \
                            goto error;                                            \
                        }                                                          \
                    }

#define SAMPLE_RATE (44100)

static paTestData data;

int main()
{
    PaStream *stream;
    PaError err;

    err = Pa_Initialize();
    if (err != paNoError) {
	goto error;
    }

    err = Pa_OpenDefaultStream (&stream,
				0,
				2,
				paFloat32,
				SAMPLE_RATE,
				paFramesPerBufferUnspecified,
				callback,
				&data);
    if (err != paNoError) {
	printf("PortAudio error: %s\n", Pa_GetErrorText(err));
	goto error;
    }

    err = Pa_StartStream (stream);
    ERROR_CHECK;

    Pa_Sleep(2000);

    err = Pa_StopStream (stream);
    ERROR_CHECK;

    err = Pa_CloseStream (stream);
    ERROR_CHECK;

    err = Pa_Terminate();
    ERROR_CHECK;

    return 0;

    error:
        Pa_Terminate();
        fprintf (stderr, "An error occured while using the portaudio stream\n");
        fprintf (stderr, "Error number: %d\n", err);
        fprintf (stderr, "Error message: %s\n", Pa_GetErrorText(err));
        return err;
}
