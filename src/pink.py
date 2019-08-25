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

from ctypes import *

# #define PINK_MAX_RANDOM_ROWS   (30)
PINK_MAX_RANDOM_ROWS = 30

# #define PINK_RANDOM_BITS       (24)
PINK_RANDOM_BITS = 24

# #define PINK_RANDOM_SHIFT      ((sizeof(long)*8)-PINK_RANDOM_BITS)
PINK_RANDOM_SHIFT = (sizeof(c_long) * 8) - PINK_RANDOM_BITS

# type of `pink_rows[]` member of `pink_noise_t` struct
PINK_ROWS_TYPE = c_long * PINK_MAX_RANDOM_ROWS

# Struct used to track the pink noise generator state
class Pink_noise_t(Structure):
    """
    typedef struct
    {
      long      pink_rows[PINK_MAX_RANDOM_ROWS];
      long      pink_running_sum;   /* Used to optimize summing of generators. */
      int       pink_index;        /* Incremented each sample. */
      int       pink_index_mask;    /* Index wrapped by ANDing with this mask. */
      float     pink_scalar;       /* Used to scale within range of -1.0 to +1.0 */
    } pink_noise_t;
    """
    _fields_ = (("pink_rows", PINK_ROWS_TYPE),
                ("pink_running_sum", c_long),
                ("pink_index", c_int),
                ("pink_index_mask", c_int),
                ("pink_scalar", c_float))
