// PNG slice file I/O, float32 or float64 on disk (see png_slice_io.h).

#include "png_slice_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

// Complex values converted per chunk: bounds the temporary buffer to 1 MiB of doubles
#define PNG_SLICE_CHUNK ((size_t)1 << 16)

static long long file_size(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    return (long long)st.st_size;
}

int png_slice_format(const char *path, size_t nvals)
{
    long long sz = file_size(path);
    if (sz < 0) return 0;
    if (sz == (long long)nvals * 2 * PNG_SLICE_F32) return PNG_SLICE_F32;
    if (sz == (long long)nvals * 2 * PNG_SLICE_F64) return PNG_SLICE_F64;
    return -1;
}

static long long exact_sqrt(long long v)
{
    if (v <= 0) return -1;
    long long r = llround(sqrt((double)v));
    for (long long c = r - 1; c <= r + 1; c++) {
        if (c > 0 && c * c == v) return c;
    }
    return -1;
}

int png_slice_detect(const char *path, int *N, int *bytes_per_real)
{
    long long sz = file_size(path);
    if (sz < 0) return 1;
    const int types[2] = {PNG_SLICE_F32, PNG_SLICE_F64};
    for (int t = 0; t < 2; t++) {
        long long per_val = 2LL * types[t];
        if (sz % per_val != 0) continue;
        long long n = exact_sqrt(sz / per_val);
        if (n > 0) {
            if (N) *N = (int)n;
            if (bytes_per_real) *bytes_per_real = types[t];
            return 0;
        }
    }
    return 2;
}

// One reader for all four (file type, buffer type) combinations. A matching type is read
// straight into the caller's buffer; otherwise through a bounded conversion buffer.
static int read_slice(const char *path, size_t nvals, void *out, int out_bytes)
{
    int fmt = png_slice_format(path, nvals);
    if (fmt == 0) return 1;
    if (fmt < 0) return 2;
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) return 1;

    size_t done = 0;
    if (fmt == out_bytes) {
        done = fread(out, 2 * (size_t)fmt, nvals, fp);
    } else {
        void *buf = malloc(PNG_SLICE_CHUNK * 2 * (size_t)fmt);
        while (buf != NULL && done < nvals) {
            size_t want = (nvals - done < PNG_SLICE_CHUNK) ? nvals - done : PNG_SLICE_CHUNK;
            size_t got = fread(buf, 2 * (size_t)fmt, want, fp);
            for (size_t i = 0; i < 2 * got; i++) {
                double v = (fmt == PNG_SLICE_F32) ? (double)((float *)buf)[i] : ((double *)buf)[i];
                if (out_bytes == PNG_SLICE_F32) ((float *)out)[2 * done + i] = (float)v;
                else ((double *)out)[2 * done + i] = v;
            }
            done += got;
            if (got != want) break;
        }
        free(buf);
    }
    fclose(fp);
    return (done == nvals) ? 0 : 3;
}

static int write_slice(const char *path, size_t nvals, const void *in, int in_bytes, int file_bytes)
{
    if (file_bytes != PNG_SLICE_F32 && file_bytes != PNG_SLICE_F64) return 3;
    FILE *fp = fopen(path, "wb");
    if (fp == NULL) return 1;

    size_t done = 0;
    if (file_bytes == in_bytes) {
        done = fwrite(in, 2 * (size_t)file_bytes, nvals, fp);
    } else {
        void *buf = malloc(PNG_SLICE_CHUNK * 2 * (size_t)file_bytes);
        while (buf != NULL && done < nvals) {
            size_t want = (nvals - done < PNG_SLICE_CHUNK) ? nvals - done : PNG_SLICE_CHUNK;
            for (size_t i = 0; i < 2 * want; i++) {
                double v = (in_bytes == PNG_SLICE_F32) ? (double)((const float *)in)[2 * done + i]
                                                       : ((const double *)in)[2 * done + i];
                if (file_bytes == PNG_SLICE_F32) ((float *)buf)[i] = (float)v;
                else ((double *)buf)[i] = v;
            }
            size_t put = fwrite(buf, 2 * (size_t)file_bytes, want, fp);
            done += put;
            if (put != want) break;
        }
        free(buf);
    }
    int close_err = fclose(fp);
    return (done == nvals && close_err == 0) ? 0 : 3;
}

int png_slice_read_double(const char *path, size_t nvals, double *out)
{
    return read_slice(path, nvals, out, PNG_SLICE_F64);
}

int png_slice_read_float(const char *path, size_t nvals, float *out)
{
    return read_slice(path, nvals, out, PNG_SLICE_F32);
}

int png_slice_write_double(const char *path, size_t nvals, const double *in, int bytes_per_real)
{
    return write_slice(path, nvals, in, PNG_SLICE_F64, bytes_per_real);
}

int png_slice_write_float(const char *path, size_t nvals, const float *in, int bytes_per_real)
{
    return write_slice(path, nvals, in, PNG_SLICE_F32, bytes_per_real);
}

void png_format_fnl(char *buf, size_t len, double f_NL)
{
    if (f_NL == floor(f_NL) && fabs(f_NL) < 1e15) snprintf(buf, len, "%.0f", f_NL);
    else snprintf(buf, len, "%g", f_NL);
}

const char *png_slice_type_name(int bytes_per_real)
{
    if (bytes_per_real == PNG_SLICE_F32) return "float32";
    if (bytes_per_real == PNG_SLICE_F64) return "float64";
    return "unknown";
}
