#ifndef PNG_SLICE_IO_H
#define PNG_SLICE_IO_H

// ====================================================================================
// PNG SLICE FILES: Phi_gaussian_y.{y}, kernel_<shape>_y.{y}, phi_<shape>_fNL<f>_divided_y.{y}
// ====================================================================================
// Each file is one y-slice: N*N complex values as interleaved (real, imag), x fastest then z,
// so element (x, z) is at index z*N + x. No header.
//
// Two element types, told apart by the file size alone:
//   float32: N*N*8  bytes  (default for new files; ZD_PNG_FileFloat64 = 0)
//   float64: N*N*16 bytes  (ZD_PNG_FileFloat64 = 1, and every file written before this option)
// The sizes cannot be confused: N^2*8 = M^2*16 would need N^2 = 2 M^2.
// Readers accept either type and convert; writers take the type from the caller.
// ====================================================================================

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bytes per real number in a slice file
#define PNG_SLICE_F32 4
#define PNG_SLICE_F64 8

// Bytes per real number of the file at path for an N*N slice (nvals = N*N complex values):
// PNG_SLICE_F32 or PNG_SLICE_F64. 0 if the file is missing, -1 if its size fits neither.
int png_slice_format(const char *path, size_t nvals);

// N and the element type from the file size alone (for readers that do not know N).
// Returns 0 on success, 1 if missing, 2 if the size is not N*N*8 or N*N*16 for any N.
int png_slice_detect(const char *path, int *N, int *bytes_per_real);

// Reads an N*N slice (either type) into out[2*nvals] doubles.
// Returns 0 on success, 1 if it cannot be opened, 2 if the size fits neither type,
// 3 on a short read.
int png_slice_read_double(const char *path, size_t nvals, double *out);

// Same, into out[2*nvals] floats.
int png_slice_read_float(const char *path, size_t nvals, float *out);

// Writes in[2*nvals] doubles as bytes_per_real (PNG_SLICE_F32 or PNG_SLICE_F64).
// Returns 0 on success, 1 if it cannot be opened, 3 on a short write or close error.
int png_slice_write_double(const char *path, size_t nvals, const double *in, int bytes_per_real);

// Same, from in[2*nvals] floats.
int png_slice_write_float(const char *path, size_t nvals, const float *in, int bytes_per_real);

// Human-readable name of an element type: "float32", "float64" or "unknown"
const char *png_slice_type_name(int bytes_per_real);

#ifdef __cplusplus
}
#endif

#endif // PNG_SLICE_IO_H
