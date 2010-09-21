#include "medusaext.h"
#include "medusa.h"

#include <stdio.h>

#define ERROR_CHECK \
{ \
if (err < 0) { \
    PyErr_SetString(PyExc_RuntimeError, Pa_GetErrorText(err)); \
    return 0; \
} \
}

static PyMethodDef _medusaextMethods[] = {
    {"portaudio_initialize", portaudio_initialize, METH_VARARGS},
    {"portaudio_terminate",  portaudio_terminate,  METH_VARARGS},
    {"get_device_count",     get_device_count,     METH_VARARGS},
    {"play_array",           play_array,           METH_VARARGS},
    {"play_tone",            play_tone,            METH_VARARGS},
	    {"play_tone2",            play_tone2,            METH_VARARGS},
    // wrappers start here...
    {"wrap_Pa_GetVersion",        wrap_Pa_GetVersion,        METH_VARARGS},
    {"wrap_Pa_GetVersionText",    wrap_Pa_GetVersionText,    METH_VARARGS},
    {"wrap_Pa_GetErrorText",      wrap_Pa_GetErrorText,      METH_VARARGS},
    {"wrap_Pa_GetHostApiCount",   wrap_Pa_GetHostApiCount,   METH_VARARGS},
    {"wrap_Pa_Initialize",        wrap_Pa_Initialize,        METH_VARARGS},
    {"wrap_Pa_Terminate",         wrap_Pa_Terminate,         METH_VARARGS},
    {"wrap_Pa_GetHostApiCount",   wrap_Pa_GetHostApiCount,   METH_VARARGS},
    {"wrap_Pa_GetDefaultHostApi", wrap_Pa_GetDefaultHostApi, METH_VARARGS},
    {"wrap_Pa_GetHostApiInfo", wrap_Pa_GetHostApiInfo, METH_VARARGS},
    {"wrap_Pa_HostApiTypeIdToHostApiIndex", wrap_Pa_HostApiTypeIdToHostApiIndex, METH_VARARGS},
    {"wrap_Pa_HostApiDeviceIndexToDeviceIndex", wrap_Pa_HostApiDeviceIndexToDeviceIndex, METH_VARARGS},
    {"wrap_Pa_GetLastHostErrorInfo", wrap_Pa_GetLastHostErrorInfo, METH_VARARGS},
    {"wrap_Pa_GetDeviceCount", wrap_Pa_GetDeviceCount, METH_VARARGS},
    {"wrap_Pa_GetDefaultInputDevice", wrap_Pa_GetDefaultInputDevice, METH_VARARGS},
    {"wrap_Pa_GetDefaultOutputDevice", wrap_Pa_GetDefaultOutputDevice, METH_VARARGS},
    {"wrap_Pa_GetDeviceInfo", wrap_Pa_GetDeviceInfo, METH_VARARGS},
    {"wrap_Pa_IsFormatSupported", wrap_Pa_IsFormatSupported, METH_VARARGS},
    // {"wrap_Pa_OpenStream", wrap_Pa_OpenStream, METH_VARARGS},
    // {"wrap_Pa_OpenDefaultStream", wrap_Pa_OpenDefaultStream, METH_VARARGS},
    // {"wrap_Pa_CloseStream", wrap_Pa_CloseStream, METH_VARARGS},
    // {"wrap_Pa_SetStreamFinishedCallback", wrap_Pa_SetStreamFinishedCallback, METH_VARARGS},
    // {"wrap_Pa_StartStream", wrap_Pa_StartStream, METH_VARARGS},
    // {"wrap_Pa_StopStream", wrap_Pa_StopStream, METH_VARARGS},
    // {"wrap_Pa_AbortStream", wrap_Pa_AbortStream, METH_VARARGS},
    // {"wrap_Pa_IsStreamStopped", wrap_Pa_IsStreamStopped, METH_VARARGS},
    // {"wrap_Pa_IsStreamActive", wrap_Pa_IsStreamActive, METH_VARARGS},
    // {"wrap_Pa_GetStreamInfo", wrap_Pa_GetStreamInfo, METH_VARARGS},
    // {"wrap_Pa_GetStreamTime", wrap_Pa_GetStreamTime, METH_VARARGS},
    // {"wrap_Pa_GetStreamCpuLoad", wrap_Pa_GetStreamCpuLoad, METH_VARARGS},
    // {"wrap_Pa_ReadStream", wrap_Pa_ReadStream, METH_VARARGS},
    // {"wrap_Pa_WriteStream", wrap_Pa_WriteStream, METH_VARARGS},
    // {"wrap_Pa_GetStreamReadAvailable", wrap_Pa_GetStreamReadAvailable, METH_VARARGS},
    // {"wrap_Pa_GetStreamWriteAvailable", wrap_Pa_GetStreamWriteAvailable, METH_VARARGS},
    // {"wrap_Pa_GetSampleSize", wrap_Pa_GetSampleSize, METH_VARARGS},
    // {"wrap_Pa_Sleep", wrap_Pa_Sleep, METH_VARARGS},
    {NULL, NULL}
};

void init_medusaext ()
{
    (void) Py_InitModule("_medusaext", _medusaextMethods);
    import_array();
}

static PyObject *
portaudio_initialize (PyObject *self, PyObject *args)
{
    PaError err;

    err = Pa_Initialize();
    ERROR_CHECK;

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
portaudio_terminate (PyObject *self, PyObject *args)
{
    PaError err;

    err = Pa_Terminate();
    if (err) {
        PyErr_SetString(PyExc_RuntimeError, Pa_GetErrorText(err));
        return 0;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
get_device_count (PyObject *self, PyObject *args)
{
    PaError err;
    int device_count;

    err = Pa_GetDeviceCount();
    if (err < 0) { // `PaError` is always a negative integer
        PyErr_SetString(PyExc_RuntimeError, Pa_GetErrorText(err));
        return 0;
    }
    else {
        device_count = err;
    }

    return Py_BuildValue("i", device_count);
}

static PyObject *
play_array (PyObject *self, PyObject *args)
{
    PyArrayObject *x;
    PyArrayObject *x_contig;
    double fs;

    PaStream *stream;
    PaError err;

    PaDeviceInfo* dev_info;

    PaHostApiIndex hai;
    PaDeviceIndex devi;

	int channels;

    /*
    PaStreamParameters out_stream_params;

    hai = Pa_HostApiTypeIdToHostApiIndex(paASIO);
    devi = Pa_HostApiDeviceIndexToDeviceIndex(hai, 0);

    dev_info = Pa_GetDeviceInfo(devi);
    printf("%s\n", dev_info->name);
    printf("%f\n", dev_info->defaultSampleRate);
    printf("%d\n", dev_info->maxOutputChannels);


    out_stream_params.device = devi;
    out_stream_params.channelCount = 1;
    out_stream_params.sampleFormat = paFloat32;
    out_stream_params.suggestedLatency = dev_info->defaultLowOutputLatency;
    out_stream_params.hostApiSpecificStreamInfo = NULL;
*/

    // http://docs.python.org/c-api/arg.html
    if (!PyArg_ParseTuple(args, "O!d",
                          &PyArray_Type, &x,
                          &fs)) {
        return NULL;
    }

    channels = PyArray_DIM(x, 1);
	//printf("channels: %d\n", channels);

    x_contig = (PyArrayObject *) PyArray_ContiguousFromObject(x, PyArray_DOUBLE, 1, 10);
////////////////////////////////////////////////////////////////

    err = Pa_Initialize();
    ERROR_CHECK;

    //Pa_Sleep(100);
/*
    err = Pa_OpenStream (&stream,
                         NULL,
                         &out_stream_params,
                         fs,
                         paFramesPerBufferUnspecified,
                         paNoFlag,
                         callback_play_array,
                         x);
*/

    err = Pa_OpenDefaultStream (&stream,
                                0,
                                channels,
                                paFloat32,
                                fs,
                                paFramesPerBufferUnspecified,
                                callback_play_array,
                                x_contig);
    ERROR_CHECK;

    //printf("okay in medusa\n");

    err = Pa_StartStream (stream);
    ERROR_CHECK;

    //printf("stream started\n");

    while (err = Pa_IsStreamActive(stream)) {;}

    err = Pa_CloseStream (stream);
    ERROR_CHECK;

    err = Pa_Terminate();
    ERROR_CHECK;

////////////////////////////////////////////////////////////////
    // Py_DECREF(x_contig);
    Py_INCREF(Py_None);
    return Py_None; // Return `None : NoneType` value
}

static PyObject *
play_tone (PyObject *self, PyObject *args)
{
    PaStream *stream;
    PaError err;

    //err = Pa_Initialize();

    err = Pa_OpenDefaultStream (&stream,
                                0,
                                1,
                                paFloat32,
                                200,
                                17146,
                                callback_play_tone,
                                NULL);
    ERROR_CHECK;



    err = Pa_StartStream (stream);
    ERROR_CHECK;

    //while (err = Pa_IsStreamActive(stream));

    //err = Pa_CloseStream (stream);
    //ERROR_CHECK;

    //err = Pa_Terminate();
    //ERROR_CHECK;

////////////////////////////////////////////////////////////////

    Py_INCREF(Py_None);
    return Py_None; // Return `None : NoneType` value
}

static PyObject *
play_tone2 (PyObject *self, PyObject *args)
{
    PaStream *stream;
    PaError err;

    //err = Pa_Initialize();

    err = Pa_OpenDefaultStream (&stream,
                                0,
                                1,
                                paFloat32,
                                44100,
                                17146,
                                callback_play_tone2,
                                NULL);
    ERROR_CHECK;



    err = Pa_StartStream (stream);
    ERROR_CHECK;

    //while (err = Pa_IsStreamActive(stream));

    //err = Pa_CloseStream (stream);
    //ERROR_CHECK;

    //err = Pa_Terminate();
    //ERROR_CHECK;

////////////////////////////////////////////////////////////////

    Py_INCREF(Py_None);
    return Py_None; // Return `None : NoneType` value
}


// Start of Portaudio wrappers...

static PyObject *
wrap_Pa_GetVersion (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#66da08bcf908e0849c62a6b47f50d7b4
{
    return Py_BuildValue("i", Pa_GetVersion());
}

static PyObject *
wrap_Pa_GetVersionText (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#28f3fd9e6d9f933cc695abea71c4b445
{
    const char *version_text;

    version_text = Pa_GetVersionText();

    return Py_BuildValue("s", version_text);
}

static PyObject *
wrap_Pa_GetErrorText (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#e606855a611cf29c7d2d7421df5e3b5d
{
    PaError err;
    if (!PyArg_ParseTuple(args, "i",
                          &err)) {
        return NULL;
    }
    return Py_BuildValue("s", Pa_GetErrorText(err));
}

static PyObject *
wrap_Pa_Initialize (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#bed859482d156622d9332dff9b2d89da
{
    PaError err;

    err = Pa_Initialize();
    ERROR_CHECK;

    return Py_BuildValue("i", err);
}

static PyObject *
wrap_Pa_Terminate (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#0db317604e916e8bd6098e60e6237221
{
    PaError err;

    err = Pa_Terminate();
    ERROR_CHECK;

    return Py_BuildValue("i", err);
}

static PyObject *
wrap_Pa_GetHostApiCount (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#19dbdb7c8702e3f4bfc0cdb99dac3dd9
{
    PaError err;
    PaHostApiIndex host_api_count;

    err = Pa_GetHostApiCount();
    ERROR_CHECK;

    host_api_count = err;

    return Py_BuildValue("i", host_api_count);
}

static PyObject *
wrap_Pa_GetDefaultHostApi (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#e55c77f9b7e3f8eb301a6f1c0e2347ac
{
    PaError err;
    PaHostApiIndex host_api_index;

    err = Pa_GetDefaultHostApi();
    ERROR_CHECK;

    host_api_index = err;

    return Py_BuildValue("i", host_api_index);
}

static PyObject *
wrap_Pa_GetHostApiInfo (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#7c650aede88ea553066bab9bbe97ea90
{
    PaHostApiIndex host_api_index;
    const PaHostApiInfo *host_api_info;

    // Members of `struct PaHostApiInfo`
    int struct_version;
    PaHostApiTypeId type;
    const char *name;
    int device_count;
    PaDeviceIndex default_input_device;
    PaDeviceIndex default_output_device;

    if (!PyArg_ParseTuple(args, "i",
                          &host_api_index)) {
        return NULL;
    }
    host_api_info = Pa_GetHostApiInfo(host_api_index);

    if (host_api_info == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "HostApiIndex out of range");
        return 0;
    }

    struct_version = host_api_info->structVersion;
    type = host_api_info->type;
    name = host_api_info->name;
    device_count = host_api_info->deviceCount;
    default_input_device = host_api_info->defaultInputDevice;
    default_output_device = host_api_info->defaultOutputDevice;

    return Py_BuildValue("{s:i,s:i,s:s,s:i,s:i,s:i}",
                         "struct_version", struct_version,
                         "type", type,
                         "name", name,
                         "device_count", device_count,
                         "default_input_device", default_input_device,
                         "default_output_device", default_output_device);
}

static PyObject *
wrap_Pa_HostApiTypeIdToHostApiIndex (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#081c3975126d20b4226facfb7ba0620f
{
    PaError err;
    PaHostApiTypeId host_api_type_id;
    PaHostApiIndex host_api_index;

    if (!PyArg_ParseTuple(args, "i",
                          &host_api_type_id)) {
        return NULL;
    }
    err = Pa_HostApiTypeIdToHostApiIndex(host_api_type_id);
    ERROR_CHECK;

    host_api_index = err;

    return Py_BuildValue("i", host_api_index);
}

static PyObject *
wrap_Pa_HostApiDeviceIndexToDeviceIndex (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#54f306b5e5258323c95a27c5722258cd
{
    PaError err;

    PaHostApiIndex host_api;
    int host_api_device_index;

    PaDeviceIndex device_index;

    if (!PyArg_ParseTuple(args, "ii",
                          &host_api,
                          &host_api_device_index)) {
        return NULL;
    }
    err = Pa_HostApiDeviceIndexToDeviceIndex(host_api, host_api_device_index);
    ERROR_CHECK;

    device_index = err;

    return Py_BuildValue("i", device_index);
}

static PyObject *
wrap_Pa_GetLastHostErrorInfo (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#ad573f208b60577f21d2777a7c5054e0
{
    const PaHostErrorInfo* host_error_info;

    PaHostApiTypeId host_api_type;
    long error_code;
    const char *error_text;

    host_error_info = Pa_GetLastHostErrorInfo();
    if (host_error_info == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Could not get pointer to host_error_info");
        return 0;
    }

    return Py_BuildValue("{s:i,s:i,s:s}",
                         "host_api_type", host_api_type,
                         "error_code", error_code,
                         "error_text", error_text);
}

static PyObject *
wrap_Pa_GetDeviceCount (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#cfe4d3c5ec1a343f459981bfa2057f8d
{
    PaError err;
    PaDeviceIndex device_count;

    err = Pa_GetDeviceCount();
    ERROR_CHECK;

    device_count = err;

    return Py_BuildValue("i", device_count);
}

static PyObject *
wrap_Pa_GetDefaultInputDevice (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#bf9f2f82da95553d5adb929af670f74b
{
    PaError err;
    PaDeviceIndex device_index;

    err = Pa_GetDefaultInputDevice();
    ERROR_CHECK;

    device_index = err;

    return Py_BuildValue("i", device_index);
}

static PyObject *
wrap_Pa_GetDefaultOutputDevice (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#dc955dfab007624000695c48d4f876dc
{
    PaError err;
    PaDeviceIndex device_index;

    err = Pa_GetDefaultOutputDevice();
    ERROR_CHECK;

    device_index = err;

    return Py_BuildValue("i", device_index);
}

static PyObject *
wrap_Pa_GetDeviceInfo (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#c7d8e091ffc1d1d4a035704660e117eb
{
    const PaDeviceInfo* device_info;

    // Locals for members of `struct PaDeviceInfo`
    int struct_version;
    const char *name;
    PaHostApiIndex host_api;
    int max_input_channels;
    int max_output_channels;
    PaTime default_low_input_latency;
    PaTime default_low_output_latency;
    PaTime default_high_input_latency;
    PaTime default_high_output_latency;
    double default_sample_rate;

    PaDeviceIndex device;

    if (!PyArg_ParseTuple(args, "i",
                          &device)) {
        return NULL;
    }
    device_info = Pa_GetDeviceInfo(device);
    if (device_info == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Device parameter is out of range");
        return 0;
    }

    struct_version = device_info->structVersion;
    name = device_info->name;
    host_api = device_info->hostApi;
    max_input_channels = device_info->maxInputChannels;
    max_output_channels = device_info->maxOutputChannels;
    default_low_input_latency = device_info->defaultLowInputLatency;
    default_low_output_latency = device_info->defaultLowOutputLatency;
    default_high_input_latency = device_info->defaultHighInputLatency;
    default_high_output_latency = device_info->defaultHighOutputLatency;
    default_sample_rate = device_info->defaultSampleRate;

    return Py_BuildValue("{s:i,s:s,s:i,s:i,s:i,s:d,s:d,s:d,s:d,s:d}",
                         "struct_version", struct_version,
                         "name", name,
                         "host_api", host_api,
                         "max_input_channels", max_input_channels,
                         "max_output_channels", max_output_channels,
                         "default_low_input_latency", default_low_input_latency,
                         "default_low_output_latency", default_low_output_latency,
                         "default_high_input_latency", default_high_input_latency,
                         "default_high_output_latency", default_high_output_latency,
                         "default_sample_rate", default_sample_rate);
}

static PyObject *
wrap_Pa_IsFormatSupported (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#bdb313743d6efef26cecdae787a2bd3d
{
    PaError err;

    PaStreamParameters input_parameters;
    PaStreamParameters output_parameters;
    double sample_rate;

    PyObject *input_params;
    PyObject *output_params;

    if (!PyArg_ParseTuple(args, "OOd",
                          &input_params,
                          &output_params,
                          &sample_rate)) {
        return NULL;
    }

    input_parameters.device = *((PaDeviceIndex *) PyDict_GetItemString(input_params, "device"));
    input_parameters.channelCount = *((int *) PyDict_GetItemString(input_params, "channel_count"));
    input_parameters.sampleFormat = *((PaSampleFormat *) PyDict_GetItemString(input_params, "sample_format"));
    input_parameters.suggestedLatency = *((PaTime *) PyDict_GetItemString(input_params, "suggested_latency"));
    //input_parameters.hostApiSpecificStreamInfo = ((void *) PyDict_GetItemString(input_params, "host_api_specific_stream"));
    input_parameters.hostApiSpecificStreamInfo = NULL; // cop-out

    output_parameters.device = *((PaDeviceIndex *) PyDict_GetItemString(output_params, "device"));
    output_parameters.channelCount = *((int *) PyDict_GetItemString(output_params, "channel_count"));
    output_parameters.sampleFormat = *((PaSampleFormat *) PyDict_GetItemString(output_params, "sample_format"));
    output_parameters.suggestedLatency = *((PaTime *) PyDict_GetItemString(output_params, "suggested_latency"));
    //output_parameters.hostApiSpecificStreamInfo = ((void *) PyDict_GetItemString(output_params, "host_api_specific_stream"));
    output_parameters.hostApiSpecificStreamInfo = NULL;  // cop-out

    err = Pa_IsFormatSupported(&input_parameters, &output_parameters, sample_rate);
    ERROR_CHECK;

    if (err == paFormatIsSupported) {
        return Py_BuildValue("i", paFormatIsSupported);
    }
    else {
        PyErr_SetString(PyExc_RuntimeError, "Unknown runtime error in function `wrap_Pa_IsFormatSupported`");
        return 0;
    }
}

static PyObject *
wrap_Pa_OpenStream (PyObject *self, PyObject *args)
// http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#443ad16338191af364e3be988014cbbe
{
}

static PyObject *
wrap_Pa_OpenDefaultStream (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_CloseStream (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_SetStreamFinishedCallback (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_StartStream (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_StopStream (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_AbortStream (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_IsStreamStopped (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_IsStreamActive (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_GetStreamInfo (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_GetStreamTime (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_GetStreamCpuLoad (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_ReadStream (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_WriteStream (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_GetStreamReadAvailable (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_GetStreamWriteAvailable (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_GetSampleSize (PyObject *self, PyObject *args)
{
}

static PyObject *
wrap_Pa_Sleep (PyObject *self, PyObject *args)
{
}
