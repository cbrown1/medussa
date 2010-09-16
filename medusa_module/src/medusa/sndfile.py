from ctypes import *

class SF_INFO (Structure):
    """
    See: http://www.mega-nerd.com/libsndfile/api.html#open

    typedef struct {
        sf_count_t  frames ;
        int         samplerate ;
        int         channels ;
        int         format ;
        int         sections ;
        int         seekable ;
    } SF_INFO ;
    """
    _fields_ = ((frames,     c_uint),
                (samplerate, c_int),
                (channels,   c_int),
                (format,     c_int),
                (sections,   c_int),
                (seekable,   c_int))