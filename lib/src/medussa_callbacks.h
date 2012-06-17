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

#include "medussa_types.h"
#include <math.h>
#include "randomkit.h"

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

int callback_white (const void *,
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
