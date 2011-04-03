# The format field in the above SF_INFO structure is made up of the bit-wise OR
# of a major format type (values between 0x10000 and 0x08000000), a minor format
# type (with values less than 0x10000) and an optional endian-ness value. The
# currently understood formats are listed in sndfile.h as follows and also include
# bitmasks for separating major and minor file types. Not all combinations of
# endian-ness and major and minor file types are valid.
#
# [ http://www.mega-nerd.com/libsndfile/api.html#open ]

class sf_format_descriptions():

    # Major formats.

    SF_FORMAT_WAV          = "Microsoft WAV format (little endian)"
    SF_FORMAT_AIFF         = "Apple/SGI AIFF format (big endian)"
    SF_FORMAT_AU           = "Sun/NeXT AU format (big endian)"
    SF_FORMAT_RAW          = "RAW PCM data"
    SF_FORMAT_PAF          = "Ensoniq PARIS file format"
    SF_FORMAT_SVX          = "Amiga IFF / SVX8 / SV16 format"
    SF_FORMAT_NIST         = "Sphere NIST format"
    SF_FORMAT_VOC          = "VOC files"
    SF_FORMAT_IRCAM        = "Berkeley/IRCAM/CARL"
    SF_FORMAT_W64          = "Sonic Foundry's 64 bit RIFF/WAV"
    SF_FORMAT_MAT4         = "Matlab (tm) V4.2 / GNU Octave 2.0"
    SF_FORMAT_MAT5         = "Matlab (tm) V5.0 / GNU Octave 2.1"
    SF_FORMAT_PVF          = "Portable Voice Format"
    SF_FORMAT_XI           = "Fasttracker 2 Extended Instrument"
    SF_FORMAT_HTK          = "HMM Tool Kit format"
    SF_FORMAT_SDS          = "Midi Sample Dump Standard"
    SF_FORMAT_AVR          = "Audio Visual Research"
    SF_FORMAT_WAVEX        = "MS WAVE with WAVEFORMATEX"
    SF_FORMAT_SD2          = "Sound Designer 2"
    SF_FORMAT_FLAC         = "FLAC lossless file format"
    SF_FORMAT_CAF          = "Core Audio File format"
    SF_FORMAT_WVE          = "Psion WVE format"
    SF_FORMAT_OGG          = "Xiph OGG container"
    SF_FORMAT_MPC2K        = "Akai MPC 2000 sampler"
    SF_FORMAT_RF64         = "RF64 WAV file"

    # Subtypes from here on.

    SF_FORMAT_PCM_S8       = "Signed 8 bit data"
    SF_FORMAT_PCM_16       = "Signed 16 bit data"
    SF_FORMAT_PCM_24       = "Signed 24 bit data"
    SF_FORMAT_PCM_32       = "Signed 32 bit data"

    SF_FORMAT_PCM_U8       = "Unsigned 8 bit data (WAV and RAW only)"

    SF_FORMAT_FLOAT        = "32 bit float data"
    SF_FORMAT_DOUBLE       = "64 bit float data"

    SF_FORMAT_ULAW         = "U-Law encoded"
    SF_FORMAT_ALAW         = "A-Law encoded"
    SF_FORMAT_IMA_ADPCM    = "IMA ADPCM"
    SF_FORMAT_MS_ADPCM     = "Microsoft ADPCM"

    SF_FORMAT_GSM610       = "GSM 6.10 encoding"
    SF_FORMAT_VOX_ADPCM    = "Oki Dialogic ADPCM encoding"

    SF_FORMAT_G721_32      = "32kbs G721 ADPCM encoding"
    SF_FORMAT_G723_24      = "24kbs G723 ADPCM encoding"
    SF_FORMAT_G723_40      = "40kbs G723 ADPCM encoding"

    SF_FORMAT_DWVW_12      = "12 bit Delta Width Variable Word encoding"
    SF_FORMAT_DWVW_16      = "16 bit Delta Width Variable Word encoding"
    SF_FORMAT_DWVW_24      = "24 bit Delta Width Variable Word encoding"
    SF_FORMAT_DWVW_N       = "N bit Delta Width Variable Word encoding"

    SF_FORMAT_DPCM_8       = "8 bit differential PCM (XI only)"
    SF_FORMAT_DPCM_16      = "16 bit differential PCM (XI only)"

    SF_FORMAT_VORBIS       = "Xiph Vorbis encoding"

class sf_formats():

    # Major formats.

    SF_FORMAT_WAV          = 0x010000
    SF_FORMAT_AIFF         = 0x020000
    SF_FORMAT_AU           = 0x030000
    SF_FORMAT_RAW          = 0x040000
    SF_FORMAT_PAF          = 0x050000
    SF_FORMAT_SVX          = 0x060000
    SF_FORMAT_NIST         = 0x070000
    SF_FORMAT_VOC          = 0x080000
    SF_FORMAT_IRCAM        = 0x0A0000
    SF_FORMAT_W64          = 0x0B0000
    SF_FORMAT_MAT4         = 0x0C0000
    SF_FORMAT_MAT5         = 0x0D0000
    SF_FORMAT_PVF          = 0x0E0000
    SF_FORMAT_XI           = 0x0F0000
    SF_FORMAT_HTK          = 0x100000
    SF_FORMAT_SDS          = 0x110000
    SF_FORMAT_AVR          = 0x120000
    SF_FORMAT_WAVEX        = 0x130000
    SF_FORMAT_SD2          = 0x160000
    SF_FORMAT_FLAC         = 0X170000
    SF_FORMAT_CAF          = 0x180000
    SF_FORMAT_WVE          = 0x190000
    SF_FORMAT_OGG          = 0x200000
    SF_FORMAT_MPC2K        = 0x210000
    SF_FORMAT_RF64         = 0x220000

    # Subtypes from here on.

    SF_FORMAT_PCM_S8       = 0x0001
    SF_FORMAT_PCM_16       = 0x0002
    SF_FORMAT_PCM_24       = 0x0003
    SF_FORMAT_PCM_32       = 0x0004

    SF_FORMAT_PCM_U8       = 0x0005

    SF_FORMAT_FLOAT        = 0x0006
    SF_FORMAT_DOUBLE       = 0x0007

    SF_FORMAT_ULAW         = 0x0010
    SF_FORMAT_ALAW         = 0x0011
    SF_FORMAT_IMA_ADPCM    = 0x0012
    SF_FORMAT_MS_ADPCM     = 0x0013

    SF_FORMAT_GSM610       = 0x0020
    SF_FORMAT_VOX_ADPCM    = 0x0021

    SF_FORMAT_G721_32      = 0x0030
    SF_FORMAT_G723_24      = 0x0031
    SF_FORMAT_G723_40      = 0x0032

    SF_FORMAT_DWVW_12      = 0x0040
    SF_FORMAT_DWVW_16      = 0x0041
    SF_FORMAT_DWVW_24      = 0x0042
    SF_FORMAT_DWVW_N       = 0x0043

    SF_FORMAT_DPCM_8       = 0x0050
    SF_FORMAT_DPCM_16      = 0x0051

    SF_FORMAT_VORBIS       = 0x0060

