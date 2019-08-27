#pragma once
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MEDUSSA_LOGLEVEL_LABELS(v) \
    v(ERROR) \
    v(WARN) \
    v(INFO) \
    v(DEBUG)

typedef enum loglevel_ {
#define MEDUSSA_LOGLEVEL_ENUM(l)  l,
    MEDUSSA_LOGLEVEL_LABELS(MEDUSSA_LOGLEVEL_ENUM)
} loglevel;

void med_vlogf_(loglevel l, const char* source_file, unsigned source_line, const char* format, va_list args);
void med_logf_(loglevel l, const char* source_file, unsigned source_line, const char* format, ...);

#define med_logf(loglevel, ...) med_logf_(loglevel, __FILE__, __LINE__, __VA_ARGS__)
#define debug(...) med_logf(DEBUG, __VA_ARGS__)
#define info(...) med_logf(INFO, __VA_ARGS__)
#define warn(...) med_logf(WARN, __VA_ARGS__)
#define error(...) med_logf(ERROR, __VA_ARGS__)

#ifdef __cplusplus
}
#endif
