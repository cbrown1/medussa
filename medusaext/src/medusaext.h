#include <Python.h>
#include <numpy/arrayobject.h>
#include <portaudio.h>

static PyObject *portaudio_initialize (PyObject *self, PyObject *args);
static PyObject *portaudio_terminate  (PyObject *self, PyObject *args);
static PyObject *get_device_count     (PyObject *self, PyObject *args);
static PyObject *play_array           (PyObject *self, PyObject *args);
static PyObject *play_tone            (PyObject *self, PyObject *args);
static PyObject *play_tone2            (PyObject *self, PyObject *args);

static PyObject *wrap_Pa_GetVersion                      (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetVersionText                  (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetErrorText                    (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_Initialize                      (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_Terminate                       (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetHostApiCount                 (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetDefaultHostApi               (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetHostApiInfo                  (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_HostApiTypeIdToHostApiIndex     (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_HostApiDeviceIndexToDeviceIndex (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetLastHostErrorInfo            (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetDeviceCount                  (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetDefaultInputDevice           (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetDefaultOutputDevice          (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetDeviceInfo                   (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_IsFormatSupported               (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_OpenStream                      (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_OpenDefaultStream               (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_CloseStream                     (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_SetStreamFinishedCallback       (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_StartStream                     (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_StopStream                      (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_AbortStream                     (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_IsStreamStopped                 (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_IsStreamActive                  (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetStreamInfo                   (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetStreamTime                   (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetStreamCpuLoad                (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_ReadStream                      (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_WriteStream                     (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetStreamReadAvailable          (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetStreamWriteAvailable         (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_GetSampleSize                   (PyObject *self, PyObject *args);
static PyObject *wrap_Pa_Sleep                           (PyObject *self, PyObject *args);


/*
int 	Pa_GetVersion (void)
const char * 	Pa_GetVersionText (void)
const char * 	Pa_GetErrorText (PaError errorCode)
PaError 	Pa_Initialize (void)
PaError 	Pa_Terminate (void)
PaHostApiIndex 	Pa_GetHostApiCount (void)
PaHostApiIndex 	Pa_GetDefaultHostApi (void)
const PaHostApiInfo * 	Pa_GetHostApiInfo (PaHostApiIndex hostApi)
PaHostApiIndex 	Pa_HostApiTypeIdToHostApiIndex (PaHostApiTypeId type)
PaDeviceIndex 	Pa_HostApiDeviceIndexToDeviceIndex (PaHostApiIndex hostApi, int hostApiDeviceIndex)
const PaHostErrorInfo * 	Pa_GetLastHostErrorInfo (void)
PaDeviceIndex 	Pa_GetDeviceCount (void)
PaDeviceIndex 	Pa_GetDefaultInputDevice (void)
PaDeviceIndex 	Pa_GetDefaultOutputDevice (void)
const PaDeviceInfo * 	Pa_GetDeviceInfo (PaDeviceIndex device)
PaError 	Pa_IsFormatSupported (const PaStreamParameters *inputParameters, const PaStreamParameters *outputParameters, double sampleRate)
PaError 	Pa_OpenStream (PaStream **stream, const PaStreamParameters *inputParameters, const PaStreamParameters *outputParameters, double sampleRate, unsigned long framesPerBuffer, PaStreamFlags streamFlags, PaStreamCallback *streamCallback, void *userData)
PaError 	Pa_OpenDefaultStream (PaStream **stream, int numInputChannels, int numOutputChannels, PaSampleFormat sampleFormat, double sampleRate, unsigned long framesPerBuffer, PaStreamCallback *streamCallback, void *userData)
PaError 	Pa_CloseStream (PaStream *stream)
PaError 	Pa_SetStreamFinishedCallback (PaStream *stream, PaStreamFinishedCallback *streamFinishedCallback)
PaError 	Pa_StartStream (PaStream *stream)
PaError 	Pa_StopStream (PaStream *stream)
PaError 	Pa_AbortStream (PaStream *stream)
PaError 	Pa_IsStreamStopped (PaStream *stream)
PaError 	Pa_IsStreamActive (PaStream *stream)
const PaStreamInfo * 	Pa_GetStreamInfo (PaStream *stream)
PaTime 	Pa_GetStreamTime (PaStream *stream)
double 	Pa_GetStreamCpuLoad (PaStream *stream)
PaError 	Pa_ReadStream (PaStream *stream, void *buffer, unsigned long frames)
PaError 	Pa_WriteStream (PaStream *stream, const void *buffer, unsigned long frames)
signed long 	Pa_GetStreamReadAvailable (PaStream *stream)
signed long 	Pa_GetStreamWriteAvailable (PaStream *stream)
PaError 	Pa_GetSampleSize (PaSampleFormat format)
void 	Pa_Sleep (long msec)
*/
