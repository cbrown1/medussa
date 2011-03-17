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
