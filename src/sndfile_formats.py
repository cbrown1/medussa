# -*- coding: utf-8 -*-

# Copyright (c) 2010-2019 Christopher Brown
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

# The format field in the associated SF_INFO structure is made up of the 
# bit-wise OR of a major format type (values between 0x10000 and 0x08000000), 
# a minor format type (with values less than 0x10000) and an optional 
# endian-ness value. The currently understood formats are listed in sndfile.h 
# as follows and also include bitmasks for separating major and minor file 
# types. Not all combinations of endian-ness and major and minor file types 
# are valid.
#
# [ http://www.mega-nerd.com/libsndfile/api.html#open ]

class sndfile_formats():
    def __init__(self):
        self.sf_container_descriptions= {

            # Major formats.

            0x010000: "Microsoft WAV format (little endian)",
            0x020000: "Apple/SGI AIFF format (big endian)",
            0x030000: "Sun/NeXT AU format (big endian)",
            0x040000: "RAW PCM data",
            0x050000: "Ensoniq PARIS file format",
            0x060000: "Amiga IFF / SVX8 / SV16 format",
            0x070000: "Sphere NIST format",
            0x080000: "VOC files",
            0x0A0000: "Berkeley/IRCAM/CARL",
            0x0B0000: "Sonic Foundry's 64 bit RIFF/WAV",
            0x0C0000: "Matlab (tm) V4.2 / GNU Octave 2.0",
            0x0D0000: "Matlab (tm) V5.0 / GNU Octave 2.1",
            0x0E0000: "Portable Voice Format",
            0x0F0000: "Fasttracker 2 Extended Instrument",
            0x100000: "HMM Tool Kit format",
            0x110000: "Midi Sample Dump Standard",
            0x120000: "Audio Visual Research",
            0x130000: "MS WAVE with WAVEFORMATEX",
            0x160000: "Sound Designer 2",
            0X170000: "FLAC lossless file format",
            0x180000: "Core Audio File format",
            0x190000: "Psion WVE format",
            0x200000: "Xiph OGG container",
            0x210000: "Akai MPC 2000 sampler",
            0x220000: "RF64 WAV file",
        }

        self.sf_encoding_descriptions = {

            # Subtypes from here on.

            0x0001: "Signed 8 bit data",
            0x0002: "Signed 16 bit data",
            0x0003: "Signed 24 bit data",
            0x0004: "Signed 32 bit data",

            0x0005: "Unsigned 8 bit data (WAV and RAW only)",

            0x0006: "32 bit float data",
            0x0007: "64 bit float data",

            0x0010: "U-Law encoded",
            0x0011: "A-Law encoded",
            0x0012: "IMA ADPCM",
            0x0013: "Microsoft ADPCM",

            0x0020: "GSM 6.10 encoding",
            0x0021: "Oki Dialogic ADPCM encoding",

            0x0030: "32kbs G721 ADPCM encoding",
            0x0031: "24kbs G723 ADPCM encoding",
            0x0032: "40kbs G723 ADPCM encoding",

            0x0040: "12 bit Delta Width Variable Word encoding",
            0x0041: "16 bit Delta Width Variable Word encoding",
            0x0042: "24 bit Delta Width Variable Word encoding",
            0x0043: "N bit Delta Width Variable Word encoding",

            0x0050: "8 bit differential PCM (XI only)",
            0x0051: "16 bit differential PCM (XI only)",

            0x0060: "Xiph Vorbis encoding",
        }

        # Major formats.

        self.SF_CONTAINER_WAV          = 0x010000
        self.SF_CONTAINER_AIFF         = 0x020000
        self.SF_CONTAINER_AU           = 0x030000
        self.SF_CONTAINER_RAW          = 0x040000
        self.SF_CONTAINER_PAF          = 0x050000
        self.SF_CONTAINER_SVX          = 0x060000
        self.SF_CONTAINER_NIST         = 0x070000
        self.SF_CONTAINER_VOC          = 0x080000
        self.SF_CONTAINER_IRCAM        = 0x0A0000
        self.SF_CONTAINER_W64          = 0x0B0000
        self.SF_CONTAINER_MAT4         = 0x0C0000
        self.SF_CONTAINER_MAT5         = 0x0D0000
        self.SF_CONTAINER_PVF          = 0x0E0000
        self.SF_CONTAINER_XI           = 0x0F0000
        self.SF_CONTAINER_HTK          = 0x100000
        self.SF_CONTAINER_SDS          = 0x110000
        self.SF_CONTAINER_AVR          = 0x120000
        self.SF_CONTAINER_WAVEX        = 0x130000
        self.SF_CONTAINER_SD2          = 0x160000
        self.SF_CONTAINER_FLAC         = 0X170000
        self.SF_CONTAINER_CAF          = 0x180000
        self.SF_CONTAINER_WVE          = 0x190000
        self.SF_CONTAINER_OGG          = 0x200000
        self.SF_CONTAINER_MPC2K        = 0x210000
        self.SF_CONTAINER_RF64         = 0x220000

        # Subtypes from here on.

        self.SF_ENCODING_PCM_S8       = 0x0001
        self.SF_ENCODING_PCM_16       = 0x0002
        self.SF_ENCODING_PCM_24       = 0x0003
        self.SF_ENCODING_PCM_32       = 0x0004

        self.SF_ENCODING_PCM_U8       = 0x0005

        self.SF_ENCODING_FLOAT        = 0x0006
        self.SF_ENCODING_DOUBLE       = 0x0007

        self.SF_ENCODING_ULAW         = 0x0010
        self.SF_ENCODING_ALAW         = 0x0011
        self.SF_ENCODING_IMA_ADPCM    = 0x0012
        self.SF_ENCODING_MS_ADPCM     = 0x0013

        self.SF_ENCODING_GSM610       = 0x0020
        self.SF_ENCODING_VOX_ADPCM    = 0x0021

        self.SF_ENCODING_G721_32      = 0x0030
        self.SF_ENCODING_G723_24      = 0x0031
        self.SF_ENCODING_G723_40      = 0x0032

        self.SF_ENCODING_DWVW_12      = 0x0040
        self.SF_ENCODING_DWVW_16      = 0x0041
        self.SF_ENCODING_DWVW_24      = 0x0042
        self.SF_ENCODING_DWVW_N       = 0x0043

        self.SF_ENCODING_DPCM_8       = 0x0050
        self.SF_ENCODING_DPCM_16      = 0x0051

        self.SF_ENCODING_VORBIS       = 0x0060

    def get_sf_container(self, sf_format):
        for fmt in self.sf_container_descriptions.keys():
            if sf_format & fmt == fmt:
                return self.sf_container_descriptions[fmt]

    def get_sf_encoding(self, sf_format):
        for fmt in self.sf_encoding_descriptions.keys():
            if sf_format & fmt == fmt:
                return self.sf_encoding_descriptions[fmt]
