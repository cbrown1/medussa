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
