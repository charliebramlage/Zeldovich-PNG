// ====================================================================================
// KERNEL FILE I/O + Y-SLICE HERMITIAN HELPERS
// ====================================================================================

#include "kernel_io.h"
#include "../utils/png_slice_io.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int kernel_detect_N(const char *path)
{
    int N = -1, bytes = 0;
    return (png_slice_detect(path, &N, &bytes) == 0) ? N : -1;
}

int kernel_detect_file_bytes(const char *path)
{
    int N = -1, bytes = 0;
    return (png_slice_detect(path, &N, &bytes) == 0) ? bytes : 0;
}

// fftw_complex_t is real_t[2], so a slice is layout-compatible with interleaved real_t values.
int kernel_read_slice(const char *path, int N, fftw_complex_t *slice)
{
    size_t nvals = (size_t)N * (size_t)N;
#ifdef USE_DOUBLE_PRECISION
    int err = png_slice_read_double(path, nvals, (double *)slice);
#else
    int err = png_slice_read_float(path, nvals, (float *)slice);
#endif
    return (err == 0) ? 0 : (err == 1 ? 1 : 2);
}

int kernel_write_slice(const char *path, int N, const fftw_complex_t *slice, int bytes_per_real)
{
    size_t nvals = (size_t)N * (size_t)N;
#ifdef USE_DOUBLE_PRECISION
    int err = png_slice_write_double(path, nvals, (const double *)slice, bytes_per_real);
#else
    int err = png_slice_write_float(path, nvals, (const float *)slice, bytes_per_real);
#endif
    return (err == 0) ? 0 : (err == 1 ? 1 : 2);
}

// Element (x, z) of a [z][x] slice
#define SL(x, z) slice[(size_t)(z) * (size_t)N + (size_t)(x)]

void kernel_fix_self_conjugate_plane(int N, fftw_complex_t *slice)
{
    int Nh = N / 2;

    // Mirror the lower half onto the upper half: P[(N-x)%N, N-z] = conj(P[x, z])
    // for x in [0, N), z in [1, N/2). Reads rows z < N/2, writes rows z > N/2.
    // Rows z = 0 and z = N/2 map onto themselves and are handled below.
    for (int z = 1; z < Nh; z++) {
        for (int x = 0; x < N; x++) {
            int xm = (N - x) % N;
            SL(xm, N - z)[0] =  SL(x, z)[0];
            SL(xm, N - z)[1] = -SL(x, z)[1];
        }
    }
    // P[N - x, 0] = conj(P[x, 0]) and P[N - x, N/2] = conj(P[x, N/2]) for x in [1, N/2)
    for (int x = 1; x < Nh; x++) {
        SL(N - x, 0)[0]  =  SL(x, 0)[0];
        SL(N - x, 0)[1]  = -SL(x, 0)[1];
        SL(N - x, Nh)[0] =  SL(x, Nh)[0];
        SL(N - x, Nh)[1] = -SL(x, Nh)[1];
    }
    // Force real at DC/Nyquist corners
    SL(0, 0)[1] = 0.0;
    SL(0, Nh)[1] = 0.0;
    SL(Nh, 0)[1] = 0.0;
    SL(Nh, Nh)[1] = 0.0;
}

void kernel_zero_nyquist(int N, int y_p, fftw_complex_t *slice)
{
    int Nh = N / 2;
    if (y_p == Nh) {
        memset(slice, 0, sizeof(fftw_complex_t) * (size_t)N * (size_t)N);
        return;
    }
    for (int i = 0; i < N; i++) {
        SL(Nh, i)[0] = SL(Nh, i)[1] = 0.0;   // kx = N/2
        SL(i, Nh)[0] = SL(i, Nh)[1] = 0.0;   // kz = N/2
    }
}

#undef SL
