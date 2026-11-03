#ifndef ZD_MPI_GENERATION_H
#define ZD_MPI_GENERATION_H

// ====================================================================================
// ZD_MPI Y-SLICES GENERATION+FFT MODULE
// ====================================================================================
// Depends on: config.h, precision.h, types.h, fft/fft_setup.h, utils/verification.h
// External: PCG RNG, OpenMP, FFTW3
// ====================================================================================

#include "../config.h"
#include "../precision.h"
#include "../types.h"

// External RNG functions (from utils/rng.h)
#include "../utils/rng.h"
// zeldovich-PLT wrapper (for v15.2)
#include "../utils/zeldovich_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

// ====================================================================================
// Generate one pair of conjugate Y-slices (primary + conjugate mirror) in Fourier space
// with 2D FFT and return slices as flat arrays
// Arguments:
//      - N: Grid size
//      - global_y: Primary Y-index
//      - y_mirror: Mirror Y-index (N-y for conjugate pairs, y for self-conjugate)
//      - primary_slices: Output buffer for primary slice [narray][N][N]
//      - conjugate_slices: Output buffer for conjugate slice [narray][N][N] (same as primary if self-conjugate)
//      - narray: Number of arrays per slice
//      - plan_2d: Single-plane FFTW_PLAN_DFT_2D (N*N) in-place on stage_2d
//      - stage_2d: N*N fftw_complex_t staging buffer (aligned); reused per plane
//      - rank: MPI rank (for debug output only)
//      - ps_handle: zeldovich-PLT PowerSpectrum handle
//      - params_handle: zeldovich-PLT Parameters handle
//
// Uses OpenMP to parallelize X-Z loops within the rank (not recommended)
// Uses global_y for deterministic RNG seeding (thread-safe)
// ====================================================================================

void generate_zd_mpi_slice_pair_local(
    int N,
    int global_y,
    int y_mirror,
    fftw_complex_t *primary_slices,   // Flat array (all narray arrays for primary slice)
    fftw_complex_t *conjugate_slices, // Flat array (all narray arrays for conjugate slice)
    int narray,                       // Number of arrays per slice
    fftw_plan_t plan_2d,             // 2D FFT plan (single N*N plane)
    fftw_complex_t *stage_2d,       // Staging buffer N*N (aligned)
    int rank,                         // MPI rank (for debug output)
    PowerSpectrumHandle ps_handle,   // zeldovich-PLT PowerSpectrum handle
    ParametersHandle params_handle,  // zeldovich-PLT Parameters handle
    PowerSpectrumHandle ps_handle_primordial_dimensional, // primordial P(k) handle (NULL if unavailable)
    void** thread_rng_buffers);       // Pre-allocated RNG buffers [nthreads] (NULL = use malloc)

// Charlie: path prefix of the LOAD_WHITE_NOISE_FROM_FILE input (the file is {prefix}_y.{y}).
// Exactly one nonzero ZD_f_NL_<shape>:  {png_dir}/phi_<shape>_fNL<ZD_f_NL>_divided, with ZD_f_NL
//   written as an integer when whole (fNL100, fNL1000000) and with %g otherwise (fNL-2.5)
//   (the name LOAD_D_FROM_FILE writes). All shapes 0: {png_dir}/<ZD_white_noise_name>
//   (an absolute name is used as-is).
// Returns 0 on success; nonzero with a message in err if several shapes are set, no shape and
// no name is set, or a shape and a name are both set.
int zd_white_noise_path_prefix(ParametersHandle params_handle, char *buf, size_t len,
                               char *err, size_t errlen);

// Print accumulated PTimerWall breakdown for Stage 1 sub-phases
// (generation vs FFT), then reset timers. Call once after the batch loop.
void print_zd_mpi_gen_timers(int rank);

#ifdef __cplusplus
}
#endif

#endif
