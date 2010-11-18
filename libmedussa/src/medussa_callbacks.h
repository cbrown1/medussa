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