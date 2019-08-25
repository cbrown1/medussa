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

# #define RK_STATE_LEN 624
RK_STATE_LEN = 624

# type of `key[]` member of `rk_state` struct
KEY_ARR_TYPE = c_ulong * RK_STATE_LEN

# Struct used to track the PRNG state
class Rk_state(Structure):
    """
    typedef struct rk_state_
    {
        unsigned long key[RK_STATE_LEN];
        int pos;
        int has_gauss; /* !=0: gauss contains a gaussian deviate */
        double gauss;

        /* The rk_state structure has been extended to store the following
         * information for the binomial generator. If the input values of n or p
         * are different than nsave and psave, then the other parameters will be
         * recomputed. RTK 2005-09-02 */

        int has_binomial; /* !=0: following parameters initialized for
                                  binomial */
        double psave;
        long nsave;
        double r;
        double q;
        double fm;
        long m;
        double p1;
        double xm;
        double xl;
        double xr;
        double c;
        double laml;
        double lamr;
        double p2;
        double p3;
        double p4;

    }
    rk_state;
    """
    _fields_ = (("key", KEY_ARR_TYPE),
                ("pos", c_int),
                ("has_gauss", c_int),
                ("gauss", c_double),
                ("has_binomial", c_int),
                ("psave", c_double),
                ("nsave", c_long),
                ("r", c_double),
                ("q", c_double),
                ("fm", c_double),
                ("m", c_long),
                ("p1", c_double),
                ("xm", c_double),
                ("xl", c_double),
                ("xr", c_double),
                ("c", c_double),
                ("laml", c_double),
                ("lamr", c_double),
                ("p2", c_double),
                ("p3", c_double),
                ("p4", c_double))
