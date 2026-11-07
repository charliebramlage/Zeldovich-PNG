#ifndef KERNEL_IO_H
#define KERNEL_IO_H

// ====================================================================================
// KERNEL FILE I/O + Y-SLICE HERMITIAN HELPERS
// ====================================================================================
// File format (same as the DUMP_GAUSSIAN_D dump ZD_PNG_DIR/Phi_gaussian_y.{y} and the
// LOAD_D_FROM_FILE input ZD_PNG_DIR/kernel_y.{y}):
// N*N complex values, interleaved (real, imag), x fastest then z,
// so element (x, z) is at z*N + x. Slices are held in memory in the same [z][x] order.
// Files are float32 or float64, told apart by size (see ../utils/png_slice_io.h). Reads accept
// either; writes use the type the caller passes (ZD_PNG_FileFloat64). Values are converted
// to/from real_t as needed.
// ====================================================================================

#include "precision.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns N from the size of a slice file, or -1 if missing / not N*N complex float32 or float64
int kernel_detect_N(const char *path);

// Bytes per real number of a slice file (4 or 8), or 0 if missing / not a slice file
int kernel_detect_file_bytes(const char *path);

// Reads one N*N slice; returns 0 on success
int kernel_read_slice(const char *path, int N, fftw_complex_t *slice);

// Writes one N*N slice as bytes_per_real (4 = float32, 8 = float64); returns 0 on success
int kernel_write_slice(const char *path, int N, const fftw_complex_t *slice, int bytes_per_real);

// In-plane Hermitian fix-ups for the self-conjugate planes y=0 and y=N/2,
// identical to kernel_MPI.py (mirror z, then x along z=0 and z=N/2, then real corners)
void kernel_fix_self_conjugate_plane(int N, fftw_complex_t *slice);

// Zeroes every mode with kx = N/2 or kz = N/2 (and the whole plane if y_p = N/2)
void kernel_zero_nyquist(int N, int y_p, fftw_complex_t *slice);

#ifdef __cplusplus
}
#endif

#endif
