#include "log.h"
#include <cstdio>
#include <cassert>

namespace medussa { namespace {
using std::fprintf;
using std::vfprintf;

#ifdef NDEBUG
loglevel level = ERROR;
#else
loglevel level = DEBUG;
#endif

const char* const level_labels[] = {
#define STRINGIFY(l) #l,
    MEDUSSA_LOGLEVEL_LABELS(STRINGIFY)
};

extern "C" void med_vlogf_(loglevel l, const char* source, unsigned line, const char* format, va_list va) {
    if (l <= level) {
        fprintf(stderr, "%s:%u: %s ", source, line, level_labels[l]);
        vfprintf(stderr, format, va);
        fprintf(stderr, "\n");
    }
}

extern "C" void med_logf_(loglevel l, const char* source, unsigned line, const char* format, ...) {
    va_list va;
    va_start(va, format);
    ::med_vlogf_(l, source, line, format, va);
    va_end(va);
}
}}
