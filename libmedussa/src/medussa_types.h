#include <Python.h>
#include <numpy/arrayobject.h>

#include <portaudio.h>
#include <sndfile.h>

struct stream_user_data {
    void *parent;

    void *device;
    
    PaStream *stream;
    PaStreamParameters *in_param;
    PaStreamParameters *out_param;
    double fs;
    //PaStreamCallback *callback;

    double *mix_mat;
    int mix_mat_0;
    int mix_mat_1;
    double *mute_mat;
    int mute_mat_0;
    int mute_mat_1;
    int pa_fpb;
};
typedef struct stream_user_data stream_user_data;

struct finite_user_data {
    void *parent;

    int loop;
    unsigned int cursor;
    unsigned int frames;
    double duration;
};
typedef struct finite_user_data finite_user_data;

struct array_user_data {
    void *parent;
    PyObject *self;

    double *ndarr;
    int ndarr_0;
    int ndarr_1;
};
typedef struct array_user_data array_user_data;

struct sndfile_user_data {
    void *parent;
    PyObject *self;

    SNDFILE *fin;
    char *finpath;
    SF_INFO *finfo;
};
typedef struct sndfile_user_data sndfile_user_data;