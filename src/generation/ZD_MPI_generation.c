#include "ZD_MPI_generation.h"
#include "../utils/verification.h"
#include "../utils/zeldovich_wrapper.h" 
#include "../utils/plt_eigenmodes.h" 
#include "../config.h"  
#include "../precision.h"  
#include "../PTimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <filesystem>

// Fine-grained PTimerWall accumulators for Stage 1 sub-phases.
// These accumulate wall-clock time across all Y-slice calls.
static PTimerWall pt_rng_setup(1);   // RNG buffer alloc + copy + advance to z-start
static PTimerWall pt_zloop(1);       // Z-loop computation (RNG calls + coefficient math + stores)
static PTimerWall pt_mirror(1);      // Self-conjugate mirroring post-processing
static PTimerWall pt_fft_copy_in(1); // 2D staged copy primary/conjugate -> stage
static PTimerWall pt_fft_execute(1); // 2D FFTW_EXECUTE_DFT on stage (plan_dft_2d)
static PTimerWall pt_fft_copy_out(1);// 2D staged copy stage -> primary/conjugate

// Primordial power spectrum (class_pk_primordial_dimensional.dat) is loaded once
// in zeldovich_mpi_driver.cpp -- the same way the main P(k) file is (rank-0 read +
// MPI broadcast, spline built once) -- and passed in as the
// ps_handle_primordial_dimensional parameter below, exactly like ps_handle. 
#define COPY_CACHE_SIZE_CHUNK 8

static void copy_plane_chunked(fftw_complex_t *dst, const fftw_complex_t *src, int N)
{
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < N; ++y) {
        fftw_complex_t *dst_row = dst + (size_t)y * (size_t)N;
        const fftw_complex_t *src_row = src + (size_t)y * (size_t)N;
        for (int x = 0; x < N; x += COPY_CACHE_SIZE_CHUNK) {
            const int len = (x + COPY_CACHE_SIZE_CHUNK <= N)
                ? COPY_CACHE_SIZE_CHUNK
                : (N - x); // copy remaining elements
            memcpy(dst_row + x, src_row + x, (size_t)len * sizeof(fftw_complex_t));
        }
    }
}

// ====================================================================================
// Thread-local RNG helpers for parallel z-loop
// ====================================================================================

#if PARALLELIZE_Z_LOOP
// Map (z, x) to virtual position in MAX_PPD x MAX_PPD grid (stream position in rand counts)
static inline int64_t compute_virtual_position(int z, int x, int N, int Nhalf) {
    int z_v = (z <= Nhalf) ? z : (MAX_PPD - N + z);
    int x_v = (x <= Nhalf) ? x : (MAX_PPD - N + x);
    return 2 * ((int64_t)z_v * MAX_PPD + x_v); // divide by 2 later
}

#endif

// Choose which RNG to use based on local_rng_buf
// local_rng_buf == NULL: use shared RNG (zeldovich_ps_advance_rng, zeldovich_ps_cgauss)
// local_rng_buf != NULL: use buffer RNG (zeldovich_ps_advance_rng_buffer, zeldovich_ps_cgauss_from_buffer)
static inline void get_cgauss(PowerSpectrumHandle ps_handle, ParametersHandle params_handle,
    int64_t rng_index, double kmag, int64_t* nskip, void* local_rng_buf,
    double* D_real, double* D_imag) {
    if (*nskip > 0) {
        if (local_rng_buf)
            zeldovich_ps_advance_rng_buffer(local_rng_buf, *nskip);
        else
            zeldovich_ps_advance_rng(ps_handle, params_handle, rng_index, *nskip);
        *nskip = 0;
    }
    if (local_rng_buf)
        zeldovich_ps_cgauss_from_buffer(local_rng_buf, ps_handle, kmag, D_real, D_imag);
    else
        zeldovich_ps_cgauss(ps_handle, kmag, rng_index, D_real, D_imag);
}

// ====================================================================================
// PLT eigenmode: lookup, sign-flip, normalization, and k2/(k*e) scaling.
// Returns 1 on success (use_plt=1), 0 on failure (falls back to k-vector mode).
// ====================================================================================
static inline int compute_plt_eigenmode(
    int kx, int ky, int kz, int N, double k2,
    eigenmode *e, int rank, int x, int z)
{
    int ikx = (kx < 0) ? N + kx : kx; //Charlie: this creates the index for the eigenmode lookup, since the eigenmodes are stored in a 0 to N-1 grid, we need to map negative kx, ky, kz to their positive counterparts
    int iky = (ky < 0) ? N + ky : ky; //Charlie: if ky is negative, we map it to its positive counterpart by adding N, e.g. ky=-4 N=500, then iky=496, which is the index for ky=-4 in the eigenmode lookup table
    int ikz = (kz < 0) ? N + kz : kz;
    if (ikz > N / 2) ikz = N - ikz; //Charlie: they only need half the data for z, since we have Hermitian symmetry, so we can just use the positive half of the z-axis 

    if (plt_get_eigenmode(ikx, iky, ikz, (int64_t)N, e) != 0) { //Charlie: this actually calls the value of the eigenmode from the file
        if (rank == 0 && x == 0 && z == 0) {
            fprintf(stderr, "[WARNING] Failed to get PLT eigenmode for (kx=%d, ky=%d, kz=%d), using normal computation\n",
                    kx, ky, kz);
        }
        return 0;
    }

    // Set sign of z component (real FFT only gives +kz half-space) 
    // see zeldovich.cpp line 255: ehat.vec[2] *= copysign(1, kz);
    if (kz < 0) {
        e->vec[2] = -e->vec[2]; //Charlie: if kz is negative, it just gives the negative of the +kz e_z component of the eigenvalue
    }

    // Normalize eigenvector (interpolation might not preserve |e| = 1)
    // see zeldovich.cpp lines 257-263
    double e_mag = sqrt(e->vec[0] * e->vec[0] + e->vec[1] * e->vec[1] + e->vec[2] * e->vec[2]); //Charlie: calculates the magniutde of the eigenvector
    if (e_mag > 0.0) {
        e->vec[0] /= e_mag; //Charlie: normalizes the eigenvector by dividing each component by the magnitude
        e->vec[1] /= e_mag; //e_x[k_x,k_y,k_z]\hat{e_x}
        e->vec[2] /= e_mag;
    }

    // norm = k2 / (k * e) upweights each mode by 1/(khat*ehat), see zeldovich.cpp line 266
    double k_dot_e = kx * e->vec[0] + ky * e->vec[1] + kz * e->vec[2]; //Charlie: this is \vec{k} \cdot \vec{e}, the dot product of the momentum and the eigenvector
    double norm = (k2 > 0.0 && k_dot_e != 0.0) ? k2 / k_dot_e : 0.0; //Charlie: this is the scaling factor for the eigenvector, which is k^2 / (k \cdot e), which is used to scale the eigenvector to get the correct amplitude for the mode
    if (!isfinite(norm)) norm = 0.0;

    // Scale eigenvector by norm (see zeldovich.cpp lines 268-270)
    e->vec[0] *= norm;
    e->vec[1] *= norm;
    e->vec[2] *= norm;

    return 1;
}

// ====================================================================================
// Store D, F, G, H into primary and conjugate slices (Zeldovich packing scheme).
// Primary:   D + i*F,  G + i*H,   i*F*f,  (G + i*H)*f
// Conjugate: conj(D) + i*conj(F), conj(G) + i*conj(H), i*conj(F*f), ...
// ====================================================================================
static inline void store_prim_conj( //Charlie: this function enforces Hermitian symmetry in fourier space
    fftw_complex_t *prim, fftw_complex_t *conj_buf,
    int N, int x, int z, int x_mirror, int z_mirror,
    const double *D, const double *F, const double *G, const double *H,
    double f_vel, int narray, int just_density)
{
    #define _SP(a, xx, zz) prim[(int64_t)(xx) + (N) * ((zz) + (N) * (a))] //Charlie: this formula flattens the x, z, and "a" coordinates into one index. "a" counts which xz plane you are on and "xx" and "zz" are the x and z coordinates you are at on that specific plane. N is the individual size of one plane. (The maximum value for "xx" and "zz" is N-1)
    #define _SC(a, xx, zz) conj_buf[(int64_t)(xx) + (N) * ((zz) + (N) * (a))] //Charlie: "a" tracks which of the quantities (density or displacement) you are storing. Also we're storing it in the prim and conj_buf arrays, which are the primary and conjugate slices in fourier space

    if (just_density) { //Charlie: if this flag is on, you're just computing the density filed (no displacements)
        _SP(0, x, z)[0] = D[0]; //Charlie: this stores the real part of the density field into the primary slice at x, z coordinates at plane a=0 (that are inputs into the function)
        _SP(0, x, z)[1] = D[1];

        _SC(0, x_mirror, z_mirror)[0] =  D[0]; //Charlie: this stores the conjugate of the density field into the conjugate slice at x_mirror, z_mirror coordinates at plane a=0 (that are inputs into the function)
        _SC(0, x_mirror, z_mirror)[1] = -D[1];
    } else {
        // Array 0: D + i*F = (D_re - F_im) + i*(D_im + F_re) //Charlie: This handles the Hermitian Symmetry
        _SP(0, x, z)[0] = D[0] - F[1];
        _SP(0, x, z)[1] = D[1] + F[0];
        // Array 1: G + i*H = (G_re - H_im) + i*(G_im + H_re)
        _SP(1, x, z)[0] = G[0] - H[1];
        _SP(1, x, z)[1] = G[1] + H[0];

        // conj(D) + i*conj(F) = (D_re + F_im) + i*(F_re - D_im) //Charlie: This handles the Hermitian Symmetry
        _SC(0, x_mirror, z_mirror)[0] = D[0] + F[1];
        _SC(0, x_mirror, z_mirror)[1] = F[0] - D[1];
        // conj(G) + i*conj(H) = (G_re + H_im) + i*(H_re - G_im)
        _SC(1, x_mirror, z_mirror)[0] = G[0] + H[1];
        _SC(1, x_mirror, z_mirror)[1] = H[0] - G[1];

        if (narray >= 4) {
            // i*F*f = (-F_im*f) + i*(F_re*f)
            _SP(2, x, z)[0] = -F[1] * f_vel;
            _SP(2, x, z)[1] =  F[0] * f_vel;
            // (G + i*H)*f
            _SP(3, x, z)[0] = (G[0] - H[1]) * f_vel;
            _SP(3, x, z)[1] = (G[1] + H[0]) * f_vel;

            // i*conj(F*f) = (F_im*f) + i*(F_re*f)
            _SC(2, x_mirror, z_mirror)[0] = F[1] * f_vel;
            _SC(2, x_mirror, z_mirror)[1] = F[0] * f_vel;
            // conj(G*f) + i*conj(H*f) = (G_re + H_im)*f + i*(H_re - G_im)*f
            _SC(3, x_mirror, z_mirror)[0] = (G[0] + H[1]) * f_vel;
            _SC(3, x_mirror, z_mirror)[1] = (H[0] - G[1]) * f_vel;
        }
    }

    #undef _SP
    #undef _SC
}

// ====================================================================================
// Generates one pair of conjugate Y-slices (primary + mirror) in Fourier space
// Gaussian w/ power spectrum weighting (w/ zeldovich-PLT ps_handle & params_handle)
// 2D FFT: Fourier --> real space (X,Z)
// Handles RNG nskip tracking
// ====================================================================================

void generate_zd_mpi_slice_pair_local(
    int N,
    int global_y,
    int y_mirror,
    fftw_complex_t *primary_slices,   
    fftw_complex_t *conjugate_slices, 
    int narray,                       // 4 arrays per slice, 7 C numbers
    fftw_plan_t plan_2d,
    fftw_complex_t *stage_2d,         // PPD^2 staging buffer (aligned)
    int rank,
    PowerSpectrumHandle ps_handle,   // zeldovich-PLT PowerSpectrum handle
    ParametersHandle params_handle,  // zeldovich-PLT Parameters handle
    PowerSpectrumHandle ps_handle_primordial_dimensional, // primordial P(k), loaded/normalized once in the driver (NULL if unavailable)
    void** thread_rng_buffers)        // Pre-allocated RNG buffers [nthreads] (NULL = use malloc)
{
    (void)thread_rng_buffers; // currently unused, kept for API

    // ========== DIAGNOSTIC TIMING: Function-level ==========
    double t_func_start = omp_get_wtime();
    double t_setup_end, t_zloop_end, t_verify_end, t_fft_end;
    
    // Debug: Log entry for seg fault error
    // #if DEBUG_PRINTS
    // fprintf(stderr,
    //         "[Rank %d] ENTER generate_zd_mpi_slice_pair_local: "
    //         "Y_primary=%d, Y_mirror=%d, ps_handle=%p, params_handle=%p\n",
    //         rank, global_y, y_mirror, (void*)ps_handle, (void*)params_handle);
    // fflush(stderr);
    // #endif

    // Precompute N/2 to avoid repeated division in loops
    int Nhalf = N / 2;
    
    // ========== Check for density-only mode (qdensity == 2) ==========
    // In density-only mode, skip compute F, G, H (displacements), only store D (density)
    int just_density = 0;
    if (params_handle != NULL) {
        int qdensity = zeldovich_params_get_qdensity(params_handle);
        just_density = (qdensity == 2);
    }
    
    // ========== PLT Rescaling factors (computed once per function call) ==========
    // These are used to rescale F, G, H when qPLTrescale is enabled
    // target_f: continuum linear theory growth rate
    // a_NL: scale factor at target redshift
    // a0: scale factor at initial redshift
    double target_f = 1.0;
    double a_NL = 1.0;
    double a0 = 1.0;
    int qPLTrescale = 0;
    
    // Only compute rescaling factors if we're computing displacements
    if (params_handle != NULL && !just_density) {
        double f_cluster = zeldovich_params_get_f_cluster(params_handle);
        // target_f is the continuum linear theory growth rate (fluid limit)
        target_f = (sqrt(1. + 24. * f_cluster) - 1.) * 0.25;
        
        qPLTrescale = zeldovich_params_get_qPLTrescale(params_handle);
        if (qPLTrescale) {
            double z_initial = zeldovich_params_get_z_initial(params_handle);
            double PLT_target_z = zeldovich_params_get_PLT_target_z(params_handle);
            a_NL = 1. / (1. + PLT_target_z);  // Scale factor at target redshift
            a0 = 1. / (1. + z_initial);       // Scale factor at initial redshift
        }
    }

    // ========== k_cutoff filtering parameters (computed once per function call) ==========
    // Calculate k2_cutoff for filtering high-wavenumber modes
    // See zeldovich.cpp: k2_cutoff = nyquist^2 / (k_cutoff^2)
    double k_cutoff = 1.0;  // Default value
    int CornerModes = 0;    // Default value
    double k2_cutoff = 0.0; // Will be calculated
    
    if (params_handle != NULL) {
        k_cutoff = zeldovich_params_get_k_cutoff(params_handle);
        CornerModes = zeldovich_params_get_CornerModes(params_handle);
    }

    // f_NL amplitude, read from ZD_f_NL in the .par file (default 0.0 = Gaussian)
    double f_NL = 0.0;
    if (params_handle != NULL) {
        f_NL = zeldovich_params_get_f_NL(params_handle);
    }
    
    // Calculate k2_cutoff: k2_cutoff = (N/2)^2 / (k_cutoff^2)
    // Ex. N=16, k_cutoff=1.0: k2_cutoff = 8^2 / 1.0^2 = 64
    double Nhalf_dbl = (double)Nhalf;
    k2_cutoff = (Nhalf_dbl * Nhalf_dbl) / (k_cutoff * k_cutoff);

    // Fundamental wavenumber (set once per function call)
    double fundamental = 1.0;
    if (params_handle != NULL) {
        fundamental = zeldovich_params_get_fundamental(params_handle);
    }

    // Access primary and conj slices
    #define PRIM_SLICE(array_idx, x, z) \
        primary_slices[(int64_t)(x) + (N) * ((z) + (N) * (array_idx))]
     
    #define CONJ_SLICE(array_idx, x, z) \
        conjugate_slices[(int64_t)(x) + (N) * ((z) + (N) * (array_idx))]
    
    // RNG verification: Track total RNG calls and skips
    #if VERIFY_RNG_CALLS
    int64_t total_rng_calls = 0;  // Count of cgauss() calls (each uses 2 random numbers)
    int64_t total_rng_skips = 0;  // Count of skipped numbers (from nskip advances, including D=0 skips)
    #endif
    
    // RNG consistency: Skip random numbers for missing grid points when N < MAX_PPD
    // so we can reproduce sim boxes w/ up to MAX_PPD x MAX_PPD grid generators
    int64_t nskip = 0;
    if (N < MAX_PPD) {
        // Sequential: track skip incrementally during iteration (like zeldovich.cpp)
        nskip = 0;  // Will be accumulated during loops
    }
    
    t_setup_end = omp_get_wtime();

    //Charlie: adding a file 

#if (!!DUMP_GAUSSIAN_D + !!LOAD_D_FROM_FILE + !!LOAD_WHITE_NOISE_FROM_FILE) > 1
#error "Only one of DUMP_GAUSSIAN_D, LOAD_D_FROM_FILE, LOAD_WHITE_NOISE_FROM_FILE may be enabled at a time"
#endif


    const char *png_dir = "png_slices";
    if (params_handle != NULL) {
        const char *parsed_png_dir = zeldovich_params_get_png_dir(params_handle);
        if (parsed_png_dir != NULL && parsed_png_dir[0] != '\0') png_dir = parsed_png_dir;
    }
    {
        std::error_code png_dir_ec;
        std::filesystem::create_directories(png_dir, png_dir_ec);
        if (png_dir_ec) {
            fprintf(stderr, "ERROR: Could not create directory %s: %s\n", png_dir, png_dir_ec.message().c_str());
        }
    }

#if DUMP_GAUSSIAN_D
FILE *dump_fp = NULL;
char dump_filename[1024];
snprintf(dump_filename, sizeof(dump_filename), "%s/Phi_gaussian_y.%d", png_dir, global_y);
dump_fp = fopen(dump_filename, "wb");
if (dump_fp == NULL) {
    fprintf(stderr, "ERROR: Could not open dump file %s\n", dump_filename);
}
// File layout (no x,y,z indices stored): N*N complex values as interleaved doubles
// (real, imag), x fastest, then z. Element (x,z) lives at index z*N + x.
// Filled inside the (possibly OpenMP-parallel) z-loop, then written once after the loop,
// so the on-disk order does not depend on thread scheduling.
double *dump_buf = (double*)malloc((size_t)N * (size_t)N * 2 * sizeof(double));
if (dump_buf == NULL) {
    fprintf(stderr, "ERROR: Could not allocate dump buffer\n");
}

#endif

#if LOAD_D_FROM_FILE
    // Charlie: every f_NL shape enabled in the .par (nonzero ZD_f_NL_<shape>) is handled in one run.
    // The z-loop only stores phi_G * s and s, with s = (5/3)/sqrt(P_primordial(k)) (0 on masked modes).
    // After the z-loop each shape is done in turn, reusing one buffer:
    //     kernel_<shape>_y.{y}  ->  (phi_G + f_NL*kernel) * s  ->  phi_<shape>_divided_y.{y}
    // so memory does not grow with the number of shapes. D stays Gaussian in this run.
    // All buffers: interleaved (real, imag), x fastest, then z, as in the Phi_gaussian dump.
    static const char *fnl_shape_names[4] = {"local", "equil", "orthoCMB", "orthoLSS"};
    size_t nmodes = (size_t)N * (size_t)N;
    double *phiG_scaled = (double*)malloc(nmodes * 2 * sizeof(double));  // phi_G * s
    double *ng_scale    = (double*)malloc(nmodes * sizeof(double));      // s
    double *shape_buf   = (double*)malloc(nmodes * 2 * sizeof(double));  // kernel in, divided phi out
    if (phiG_scaled == NULL || ng_scale == NULL || shape_buf == NULL) {
        fprintf(stderr, "ERROR: Could not allocate phi_divided buffers\n");
    }
#endif


#if LOAD_WHITE_NOISE_FROM_FILE

    double *loaded_noise_real = (double*)malloc((size_t)N * (size_t)N * sizeof(double));
    double *loaded_noise_imag = (double*)malloc((size_t)N * (size_t)N * sizeof(double));
    {
        char load_filename[1024];
        snprintf(load_filename, sizeof(load_filename), "%s/phi_divided_y.%d", png_dir, global_y); 
        FILE *load_fp = fopen(load_filename, "rb");
        if (load_fp == NULL) {
            fprintf(stderr, "ERROR: Could not open %s for reading phi", load_filename);
        } else {
            // File holds N*N interleaved (real, imag) doubles, x fastest then z,
            // so element i is already at index z*N + x.
            size_t nvals = (size_t)N * (size_t)N;
            double *tmp = (double*)malloc(nvals * 2 * sizeof(double));
            size_t nread = (tmp != NULL) ? fread(tmp, 2 * sizeof(double), nvals, load_fp) : 0;
            if (nread != nvals) {
                fprintf(stderr, "ERROR: short read in %s (got %zu/%zu complex values)\n",
                        load_filename, nread, nvals);
            }
            if (tmp != NULL) {
                for (size_t i = 0; i < nread; i++) {
                    loaded_noise_real[i] = tmp[2*i];
                    loaded_noise_imag[i] = tmp[2*i + 1];
                }
                free(tmp);
            }
            fclose(load_fp);
        }
    }
#endif



    // ========== Unified z-loop: handles both conjugate-pair and self-conjugate ==========
#if PARALLELIZE_Z_LOOP
    {
    pt_rng_setup.Start(0);
    int _max_t = omp_get_max_threads();
    size_t _rng_size = zeldovich_ps_rng_buffer_size();
    void** _rng_bufs = (void**)calloc(_max_t, sizeof(void*)); // calloc: unassigned slots = NULL so free(NULL) = safe, no bad for freeing unassigned slots
    void* local_rng_buf = NULL;
    int _rng_ready = 0; // ensures we dont allocate+copy local_rng_buf for every z (only once per slice)
    int _zloop_started = 0; // choose one thread to start zloop timer (measures walltime for whole parallel region, not per thread cpu time)

    #pragma omp parallel for schedule(static) firstprivate(local_rng_buf, nskip, _rng_ready) // every thread: priv ptr to local_rng_buf, nskip, _rng_ready
    for (int z = 0; z < N; z++) {
        if (!_rng_ready) {
            int _tid = omp_get_thread_num();
            local_rng_buf = malloc(_rng_size);
            _rng_bufs[_tid] = local_rng_buf;
            // Serial zeldovich never uses v2rng[N/2]; that Y plane is zeroed post-FFT.
            // Here every mode has abs(ky)==N/2, so the z-loop only accumulates nskip (no cgauss).
            if (global_y != Nhalf) {
                zeldovich_ps_get_rng_copy(ps_handle, global_y, local_rng_buf);
                int64_t vstart = compute_virtual_position(z, 0, N, Nhalf) / 2;
                if (vstart > 0)
                    zeldovich_ps_advance_rng_buffer(local_rng_buf, vstart);
            } else {
                memset(local_rng_buf, 0, _rng_size);
            }
            _rng_ready = 1;

            int prev_value;
            #pragma omp atomic capture
            prev_value = _zloop_started++;
            if (prev_value == 0) { // walltimer: only 1 thread stops RNG_setup timer & starts zloop timer 
                pt_rng_setup.Stop(0);
                pt_zloop.Start(0);
            }
        }
#else // OMP-SERIAL VERSION
    pt_rng_setup.Start(0);
    pt_rng_setup.Stop(0);
    pt_zloop.Start(0);
    void* local_rng_buf = NULL;
    for (int z = 0; z < N; z++) {
#endif
        // RNG consistency: When crossing Nyquist boundary (z == Nhalf + 1),
        // skip ALL missing z-rows (z = N to MAX_PPD-1, each containing MAX_PPD x-values)
        // The missing frequencies are in the MIDDLE of the MAX_PPD array (high positive and negative k),
        // [0, 1, ..., N/2, -N/2+1, ..., -1]
        if (z == Nhalf + 1 && N < MAX_PPD) {
            int64_t skip_amount = (int64_t)(MAX_PPD - N) * (int64_t)MAX_PPD;
            nskip += skip_amount;
            
            #if DEBUG_RNG_SKIP
            int log_skip = (global_y <= MAX_DEBUG_COORD) || (global_y == Nhalf - 1);
            if (log_skip) {
                fprintf(stderr, "[SKIP-DEBUG] N=%d Y=%d z=%d: ACCUMULATE skip at Nyquist boundary: +%lld (missing z-rows), total nskip=%lld\n",
                        N, global_y, z, (long long)skip_amount, (long long)nskip);
                fflush(stderr);
            }
            #endif
        }
        
        // ky, kz, abs_ky, abs_kz are constant over x for this (z, global_y); compute once per z-row
        int ky = (global_y > Nhalf) ? global_y - N : global_y;
        int kz = (z > Nhalf) ? z - N : z;
        int abs_ky = (ky < 0) ? -ky : ky;
        int abs_kz = (kz < 0) ? -kz : kz;
        
        for (int x = 0; x < N; x++) {
            int x_mirror = (x == 0) ? 0 : N - x; //Charlie: x=0 case for mirroring
            int z_mirror = (z == 0) ? 0 : N - z;
            
            // ========== STEP 1: Calculate k-vector components ==========
            // RNG consistency: When crossing Nyquist boundary (x == Nhalf + 1),
            if (x == Nhalf + 1 && N < MAX_PPD) {
                int64_t skip_amount = (int64_t)(MAX_PPD - N);
                nskip += skip_amount;
                
                #if DEBUG_RNG_SKIP
                int log_skip = (z <= MAX_DEBUG_COORD && global_y <= MAX_DEBUG_COORD) ||
                               (z == Nhalf - 1 && global_y == Nhalf - 1);
                if (log_skip) {
                    fprintf(stderr, "[SKIP-DEBUG] N=%d Y=%d z=%d x=%d: ACCUMULATE skip at Nyquist boundary: +%lld (missing x-values), total nskip=%lld\n",
                            N, global_y, z, x, (long long)skip_amount, (long long)nskip);
                    fflush(stderr);
                }
                #endif
            }
            
            // kx depends on x; ky, kz, abs_ky, abs_kz already set above
            int kx = (x > Nhalf) ? x - N : x;
            int k2_int = kx*kx + ky*ky + kz*kz;
            double k2 = (double)k2_int;
            
            // ========== STEP 2: Generate D using RNG or cgauss() ==========
            // Nyquist frequency zeroing: Force Nyquist elements to zero for all three axes
            // This matches zeldovich.cpp line 354: abs(kx)==kmax || abs(kz)==kmax || abs(ky)==kmax
            int abs_kx = (kx < 0) ? -kx : kx;
            int is_nyquist = (abs_kx == Nhalf || abs_ky == Nhalf || abs_kz == Nhalf);
            
            // DC, Nyquist, and k_cutoff (corner) modes; also reused to mask the non-Gaussian field below
            int zeroed_mode = (k2 == 0.0) || is_nyquist || (!CornerModes && (double)k2_int >= k2_cutoff);

            fftw_complex D;
            fftw_complex phi_G;
                if (zeroed_mode) { //handles skipping DC mode (k=0) and Nyquist modes (k=kmax) for all three axes (and does it to keep the random number generator in sync with the serial version)
                    D[0] = D[1] = 0.0;
                    phi_G[0] = phi_G[1] = 0.0;
                    nskip++;
                    #if DEBUG_RNG_SKIP
                    int log_skip = (x <= MAX_DEBUG_COORD && global_y <= MAX_DEBUG_COORD && z <= MAX_DEBUG_COORD) ||
                                (x == Nhalf - 1 && global_y == Nhalf - 1 && z == Nhalf - 1);
                    if (log_skip) {
                        if (k2 == 0.0) {
                            fprintf(stderr, "[SKIP-DEBUG] N=%d Y=%d (x,z)=(%d,%d): D=0 (DC mode), ACCUMULATE nskip++ (total=%lld)\n",
                                    N, global_y, x, z, (long long)nskip);
                        } else if (is_nyquist) {
                            fprintf(stderr, "[SKIP-DEBUG] N=%d Y=%d (x,z)=(%d,%d): D=0 (Nyquist: kx=%d ky=%d kz=%d), ACCUMULATE nskip++ (total=%lld)\n",
                                    N, global_y, x, z, kx, ky, kz, (long long)nskip);
                        } else {
                            fprintf(stderr, "[SKIP-DEBUG] N=%d Y=%d (x,z)=(%d,%d): D=0 (k_cutoff: k2=%d >= %.1f), ACCUMULATE nskip++ (total=%lld)\n",
                                    N, global_y, x, z, k2_int, k2_cutoff, (long long)nskip);
                        }
                        fflush(stderr);
                    }
                    #endif
                } 
                else if (ps_handle != NULL && params_handle != NULL) {
                    double k2_phys = k2 * fundamental * fundamental;
                    double kmag = sqrt(k2_phys);
                    int64_t rng_index = global_y;
                    
                    double D_real, D_imag;
                    #if VERIFY_RNG_CALLS
                    if (nskip > 0 && local_rng_buf == NULL) {
                        #pragma omp atomic
                        total_rng_skips += nskip;
                    }
                    #endif
                    get_cgauss(ps_handle, params_handle, rng_index, kmag, &nskip, local_rng_buf, &D_real, &D_imag); //cgauss here returns D_real and D_imag

                    #if VERIFY_RNG_CALLS
                    #pragma omp atomic
                    total_rng_calls++;
                    #endif
                    
                    D[0] = (real_t)D_real; //Charlie: This is the Gaussian density field we want to export. I need to check if Hermitian symmetry has been handled yet
                    D[1] = (real_t)D_imag;

                    phi_G[0] = D[0];
                    phi_G[1] = D[1];
                    {
                        double k2_phys = k2 * fundamental * fundamental;
                        double kmag = sqrt(k2_phys);
                        if (kmag > 0.0) {
                            double P_density    = zeldovich_ps_power(ps_handle, kmag); //Charlie: calculates the power spectrum

                            // P_primordial comes from its own independently-
                            // loaded file/spline/handle (class_pk_primordial_dimensional.dat),
                            // evaluated with the same zeldovich_ps_power() call used for
                            // P_density above -- not inferred from ps_handle's shape.
                            double P_primordial = 0.0;
                            if (ps_handle_primordial_dimensional != NULL) {
                                P_primordial = zeldovich_ps_power(ps_handle_primordial_dimensional, kmag);
                            }

                            if (P_density > 0.0) {
                                double ratio = sqrt(P_primordial/P_density);
                                phi_G[0] = (real_t)((double)D[0] *ratio *(3.0/5.0)); 
                                phi_G[1] = (real_t)((double)D[1] *ratio *(3.0/5.0)); 
                            }
                        }
                    }
                }
                else { 
                    fprintf(stderr, "ERROR: ps_handle or params_handle is NULL for a mode "
                                    "that requires random generation (x=%d, y=%d, z=%d, k2=%g). "
                                    "This should never happen — check caller setup.\n",
                                    x, global_y, z, k2);
                    D[0] = D[1] = 0.0;   // still initialize, so we fail loudly rather than crash from UB
                    phi_G[0] = phi_G[1] = 0.0;
                    fflush(stderr);
                }


                //Dump the gaussian phi into files
                 #if DUMP_GAUSSIAN_D
                //Transform D into Phi (D itself is left untouched for downstream use)
                // Store into the slice buffer at [z*N + x]; written to disk after the z-loop.
                if (dump_buf != NULL) {
                    dump_buf[2 * ((size_t)z * (size_t)N + (size_t)x)]     = (double)phi_G[0];
                    dump_buf[2 * ((size_t)z * (size_t)N + (size_t)x) + 1] = (double)phi_G[1];
                }
 
                // Charlie: debug print — check a handful of modes as they're dumped
                if (rank == 0 && global_y == 0 && x < 3 && z < 3) {
                    fprintf(stderr, "[DUMP] (x=%d, y=%d, z=%d): D_real=%.15e, D_imag=%.15e\n",
                        x, global_y, z, (double)D[0], (double)D[1]);
                    fflush(stderr);
                    }
                if (rank == 0 && global_y == Nhalf - 1 && x < 3 && z < 3) {
                        fprintf(stderr, "[DUMP] (x=%d, y=%d, z=%d): D_real=%.15e, D_imag=%.15e\n",
                            x, global_y, z, (double)D[0], (double)D[1]);
                        fflush(stderr);
                    }  
                #endif

            #if LOAD_D_FROM_FILE
                // Charlie: s = (5/3)/sqrt(P_primordial(k)) turns phi_NG back into random numbers.
                // P_primordial(k) is read from class_pk_primordial_dimensional.dat via
                // ps_handle_primordial_dimensional, with the same k2_phys/kmag convention as the
                // cgauss branch above. s = 0 applies the DC/Nyquist/k_cutoff mask to the
                // non-Gaussian phi (phi^2 has power outside the sphere) and covers P_primordial <= 0.
                // The kernel shapes are added after the z-loop.
                {
                    double s = 0.0;
                    if (!zeroed_mode && ps_handle != NULL && ps_handle_primordial_dimensional != NULL) {
                        double k2_phys = k2 * fundamental * fundamental;
                        double kmag = sqrt(k2_phys);
                        double P_primordial = zeldovich_ps_power(ps_handle_primordial_dimensional, kmag);
                        if (P_primordial > 0.0) s = (5.0/3.0) / sqrt(P_primordial);
                    }
                    size_t i = (size_t)z * (size_t)N + (size_t)x;
                    if (ng_scale != NULL) ng_scale[i] = s;
                    if (phiG_scaled != NULL) {
                        phiG_scaled[2 * i]     = (double)phi_G[0] * s;
                        phiG_scaled[2 * i + 1] = (double)phi_G[1] * s;
                    }
                }
            #endif

            fftw_complex noise;
            #if LOAD_WHITE_NOISE_FROM_FILE
                noise[0] = (real_t)loaded_noise_real[(int64_t)x + z * N];
                noise[1] = (real_t)loaded_noise_imag[(int64_t)x + z * N];
                if (zeroed_mode) { // Charlie: same DC/Nyquist/k_cutoff mask as the Gaussian field
                    noise[0] = noise[1] = 0.0;
                }


                //Charlie: Creating D from our random numbers (from phi)
                double D_real, D_imag;
                if (ps_handle != NULL) {
                    double k2_phys = k2 * fundamental * fundamental;
                    double kmag = sqrt(k2_phys);
                    zeldovich_ps_inputted_ic(ps_handle, kmag, noise[0], noise[1], &D_real, &D_imag);
                    D[0] = (real_t)D_real; 
                    D[1] = (real_t)D_imag;
                    }
                

                    // Charlie: debug print — check the same handful of modes as they're loaded
                if (rank == 0 && global_y == 0 && x < 3 && z < 3) {
                    fprintf(stderr, "[LOAD_WHITE_NOISE] (x=%d, y=%d, z=%d): D_real=%.15e, D_imag=%.15e\n",
                            x, global_y, z, (double)D[0], (double)D[1]);
                    //fprintf(stderr, "[KERNEL] (x=%d, y=%d, z=%d): kernel_real=%.15e, kernel_imag=%.15e\n",
                    //        x, global_y, z, (double)kernel[0], (double)kernel[1]);        
                    fflush(stderr);
                }
            #else    
            #endif

            // ========== STEP 3: Compute F, G, H from D ==========
            fftw_complex F, G, H;
            int use_plt = 0; 
            eigenmode e;      
            double f = 1.0;

            if (D[0] == 0.0 && D[1] == 0.0) {
                f = 0.0;
            }
            
            if (just_density) {
                F[0] = F[1] = G[0] = G[1] = H[0] = H[1] = 0.0;
            } else if (k2 == 0.0) {
                F[0] = F[1] = G[0] = G[1] = H[0] = H[1] = 0.0;
                f = 0.0;
            } else {
            double rescale = 1.0;
            double factor;

            #if defined(VERIFY_HERMITIAN_SYMMETRY) && VERIFY_HERMITIAN_SYMMETRY == 1
            // Verification mode 1: Set F=0 and H=0 to test Hermitian symmetry
            // With F=0 and H=0, conj slices should be true conjugates of primary slices (result = pure real)
            factor = rescale / (k2 * fundamental);
            
            F[0] = 0.0;
            F[1] = 0.0;
            
            G[0] = -ky * factor * D[1];
            G[1] =  ky * factor * D[0];
            
            H[0] = 0.0;
            H[1] = 0.0;
            #elif defined(VERIFY_HERMITIAN_SYMMETRY) && VERIFY_HERMITIAN_SYMMETRY == 2
            // Verification mode 2: Compute F and H from D, then set D=0 and G=0
            // Result after 3D FFT: Array 0 and Array 1 are purely imag
            factor = rescale / (k2 * fundamental);
            
            F[0] = -kx * factor * D[1];
            F[1] =  kx * factor * D[0];
            
            G[0] = 0.0;
            G[1] = 0.0;
            
            H[0] = -kz * factor * D[1];
            H[1] =  kz * factor * D[0];
            
            D[0] = 0.0;
            D[1] = 0.0;

            #else
            // Normal operation: Compute F, G, H from D
            if (params_handle != NULL) {
                int qPLT = zeldovich_params_get_qPLT(params_handle);
                if (qPLT) {
                    use_plt = compute_plt_eigenmode(kx, ky, kz, N, k2, &e, rank, x, z);
                }
            }
            
            // Compute PLT growth rate f and rescale (before computing F, G, H)
            // f is the logarithmic derivative of the growth factor that scales velocities
            if (use_plt && params_handle != NULL) {
                double f_cluster = zeldovich_params_get_f_cluster(params_handle);
                f = (sqrt(1. + 24. * e.val * f_cluster) - 1.) * 0.25;
                
                if (qPLTrescale) {
                    double plt_f = f;
                    rescale = pow(a_NL / a0, target_f - plt_f);
                }
            }
            
            // factor = rescale / (k2 * fundamental), used for both PLT and non-PLT
            factor = rescale / (k2 * fundamental);
            
            if (use_plt) {
                #if DEBUG_EIGENVECTOR
                int debug_eigen = (rank == 0 && x <= 2 && global_y <= 2 && z <= 2);
                if (debug_eigen) {
                    fprintf(stderr, "[EIGEN-DEBUG] N=%d Y=%d (x,z)=(%d,%d) k=(%d,%d,%d): e.vec=[%.6f, %.6f, %.6f] e.val=%.6f\n",
                            N, global_y, x, z, kx, ky, kz, e.vec[0], e.vec[1], e.vec[2], e.val);
                    fflush(stderr);
                }
                #endif
                F[0] = -e.vec[0] * factor * D[1];
                F[1] =  e.vec[0] * factor * D[0];
                
                G[0] = -e.vec[1] * factor * D[1];
                G[1] =  e.vec[1] * factor * D[0];
                
                H[0] = -e.vec[2] * factor * D[1];
                H[1] =  e.vec[2] * factor * D[0];
            } else {
                F[0] = -kx * factor * D[1];
                F[1] =  kx * factor * D[0];
                
                G[0] = -ky * factor * D[1];
                G[1] =  ky * factor * D[0];
                
                H[0] = -kz * factor * D[1];
                H[1] =  kz * factor * D[0];
            }
            #endif  // VERIFY_HERMITIAN_SYMMETRY
            }  // End of else block for !just_density && k2 != 0

            #if DEBUG_RNG_CONSISTENCY
            // Per-rank log file to avoid MPI interleaving (incomplete records)
            // Match zeldovich-PLT debug output to compare D, F, G, H
            // Use #if (not #ifdef): meson always defines the macro as 0 or 1.
            {
            int ppdhalf = N / 2;
            int boundary_coord = ppdhalf - 1;
            int test_boundary = (boundary_coord >= 0 &&
                                 x == boundary_coord && global_y == boundary_coord && z == boundary_coord);
            int effective_boundary = (boundary_coord <= MAX_DEBUG_BOUNDARY_COORD) ? boundary_coord : MAX_DEBUG_BOUNDARY_COORD;
            int max_test_coord = (MAX_DEBUG_COORD > effective_boundary) ? MAX_DEBUG_COORD : effective_boundary;
            int in_test_range = (x <= max_test_coord && global_y <= max_test_coord && z <= max_test_coord);
            int should_print = 0;
            if (test_boundary) {
                should_print = 1;
            } else if (N <= DEBUG_FULL_PRINT_MAX_N) {
                should_print = in_test_range;
            } else {
                if (x <= MAX_DEBUG_COORD && global_y <= MAX_DEBUG_COORD && z <= MAX_DEBUG_COORD) {
                    should_print = 1;
                } else if (in_test_range) {
                    should_print = (x % DEBUG_SAMPLE_STRIDE == 0 &&
                                   global_y % DEBUG_SAMPLE_STRIDE == 0 &&
                                   z % DEBUG_SAMPLE_STRIDE == 0);
                }
            }
            if (should_print) {
                /* 
                fprintf(stderr, "[RNG-DEBUG] N=%ld Y=%d (x,z)=(%d,%d): k=(%d,%d,%d) k2=%.6f | "
                        "D=(%.10e,%.10e) F=(%.10e,%.10e) G=(%.10e,%.10e) H=(%.10e,%.10e)\n",
                        (long)N, global_y, x, z, kx, ky, kz, k2,
                        D[0], D[1], F[0], F[1], G[0], G[1], H[0], H[1]);
                fflush(stderr);
                */
                static FILE *rng_debug_fp = NULL;
                #if PARALLELIZE_Z_LOOP
                #pragma omp critical(rng_debug_write)
                #endif
                {
                    if (rng_debug_fp == NULL) {
                        const char *base = getenv("ZD_MPI_RNG_DEBUG_DIR");
                        char path[512];
                        if (base && base[0]) {
                            snprintf(path, sizeof(path), "%s/zd_mpi_rng_debug_rank%03d.log", base, rank);
                        } else {
                            snprintf(path, sizeof(path), "zd_mpi_rng_debug_rank%03d.log", rank);
                        }
                        rng_debug_fp = fopen(path, "a");
                    }
                    if (rng_debug_fp != NULL) {
                        fprintf(rng_debug_fp, "[RNG-DEBUG] N=%ld Y=%d (x,z)=(%d,%d): k=(%d,%d,%d) k2=%.6f | "
                                "D=(%.10e,%.10e) F=(%.10e,%.10e) G=(%.10e,%.10e) H=(%.10e,%.10e)\n",
                                (long)N, global_y, x, z, kx, ky, kz, k2,
                                D[0], D[1], F[0], F[1], G[0], G[1], H[0], H[1]);
                        fflush(rng_debug_fp);
                    }
                }
            }
            }
            #endif

            // ========== STEP 4 & 5: Store primary and conjugate slices ==========
            store_prim_conj(primary_slices, conjugate_slices,
                           N, x, z, x_mirror, z_mirror,
                           D, F, G, H, f, narray, just_density);
            
        }
        
    }
#if PARALLELIZE_Z_LOOP
    pt_zloop.Stop(0);
    for (int _t = 0; _t < _max_t; _t++)
        if (_rng_bufs[_t]) free(_rng_bufs[_t]);
    free(_rng_bufs);
    }
#else
    pt_zloop.Stop(0);
#endif

//Charlie: added to close the file pointer for dumping the Gaussian density field D
//Charlie: added to close the file pointer for dumping the Gaussian density field D
#if DUMP_GAUSSIAN_D
if (dump_fp != NULL) {
    if (dump_buf != NULL) {
        size_t nvals = (size_t)N * (size_t)N;
        size_t nwritten = fwrite(dump_buf, 2 * sizeof(double), nvals, dump_fp);
        if (nwritten != nvals) {
            fprintf(stderr, "ERROR: short write to %s (wrote %zu/%zu complex values)\n",
                    dump_filename, nwritten, nvals);
        }
    }
    fclose(dump_fp);
}
free(dump_buf);
#endif

#if LOAD_D_FROM_FILE
// Charlie: one f_NL shape at a time, all through shape_buf:
// divided = (phi_G + f_NL*kernel) * s = phiG_scaled + f_NL * s * kernel
if (phiG_scaled != NULL && ng_scale != NULL && shape_buf != NULL) {
    for (int shape = 0; shape < 4; shape++) {
        if (params_handle == NULL || zeldovich_params_get_f_NL_shape(params_handle, shape) == 0.0) continue;

        char kernel_filename[1024];
        snprintf(kernel_filename, sizeof(kernel_filename), "%s/kernel_%s_y.%d",
                 png_dir, fnl_shape_names[shape], global_y);
        FILE *kernel_fp = fopen(kernel_filename, "rb");
        if (kernel_fp == NULL) {
            fprintf(stderr, "ERROR: Could not open %s for reading kernel; skipping shape %s\n",
                    kernel_filename, fnl_shape_names[shape]);
            fflush(stderr);
            continue;
        }
        size_t nread = fread(shape_buf, 2 * sizeof(double), nmodes, kernel_fp);
        fclose(kernel_fp);
        if (nread != nmodes) {
            fprintf(stderr, "ERROR: short read in %s (got %zu/%zu complex values); skipping shape %s\n",
                    kernel_filename, nread, nmodes, fnl_shape_names[shape]);
            fflush(stderr);
            continue;
        }

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nmodes; i++) {
            double fs = f_NL * ng_scale[i];
            shape_buf[2 * i]     = phiG_scaled[2 * i]     + fs * shape_buf[2 * i];
            shape_buf[2 * i + 1] = phiG_scaled[2 * i + 1] + fs * shape_buf[2 * i + 1];
        }

        char divided_filename[1024];
        snprintf(divided_filename, sizeof(divided_filename), "%s/phi_%s_divided_y.%d",
                 png_dir, fnl_shape_names[shape], global_y);
        FILE *divided_fp = fopen(divided_filename, "wb");
        if (divided_fp == NULL) {
            fprintf(stderr, "ERROR: Could not open %s for writing divided phi\n", divided_filename);
            fflush(stderr);
            continue;
        }
        size_t nwritten = fwrite(shape_buf, 2 * sizeof(double), nmodes, divided_fp);
        if (fclose(divided_fp) != 0 || nwritten != nmodes) {
            fprintf(stderr, "ERROR: short write to %s (wrote %zu/%zu complex values)\n",
                    divided_filename, nwritten, nmodes);
            fflush(stderr);
        }
    }
}
free(phiG_scaled);
free(ng_scale);
free(shape_buf);
#endif



    // ========== Self-conjugate post-processing ==========
    // Mirror first half to second half (match zeldovich.cpp lines 555-573)
    // Only applies to self-conjugate slices (Y=0 or Y=N/2)
    pt_mirror.Start(0);
    if (y_mirror == global_y) {
        if (global_y == 0 || global_y == Nhalf) {
            for (int z = 0; z < Nhalf; z++) { //Charlie: for loop for z and x 
                int z_mirror = (z == 0) ? 0 : N - z;
                // Match zeldovich.cpp: xmax = ppdhalf for z=0, ppd for z>0
                int x_max = (z == 0 ? Nhalf : N);
                for (int x = 0; x < x_max; x++) {
                    int x_mirror = (x == 0) ? 0 : N - x;
                    #if defined(VERIFY_HERMITIAN_SYMMETRY) && VERIFY_HERMITIAN_SYMMETRY == 2
                    // Mode 2: Anti-Hermitian symmetry for purely imaginary result
                    if (x == 0 && z == 0) {
                        fprintf(stderr, "[MIRROR-DEBUG] Y=%d: Mode 2 branch executing (VERIFY_HERMITIAN_SYMMETRY=2)\n", global_y);
                        fflush(stderr);
                    }
                    for (int a = 0; a < narray; a++) {
                        PRIM_SLICE(a, x_mirror, z_mirror)[0] = CONJ_SLICE(a, x_mirror, z_mirror)[0];
                        PRIM_SLICE(a, x_mirror, z_mirror)[1] = CONJ_SLICE(a, x_mirror, z_mirror)[1];
                    }
                    #else
                    // Normal operation and Mode 1: Hermitian symmetry for purely real result
                    // Copy conj(D)+i*conj(F) from conjugate_slices to primary_slices
                    if (x == 0 && z == 0) {
#if defined(VERIFY_HERMITIAN_SYMMETRY) && VERIFY_HERMITIAN_SYMMETRY == 1
                        fprintf(stderr, "[MIRROR-DEBUG] Y=%d: Mode 1 branch executing (VERIFY_HERMITIAN_SYMMETRY=1)\n", global_y);
#elif DEBUG_PRINTS
                        fprintf(stderr, "[MIRROR-DEBUG] Y=%d: Normal operation branch executing (VERIFY_HERMITIAN_SYMMETRY not defined or != 1,2)\n", global_y);
#endif
                        fflush(stderr);
                    }
                    for (int a = 0; a < narray; a++) {
                        PRIM_SLICE(a, x_mirror, z_mirror)[0] = CONJ_SLICE(a, x_mirror, z_mirror)[0];
                        PRIM_SLICE(a, x_mirror, z_mirror)[1] = CONJ_SLICE(a, x_mirror, z_mirror)[1];
                    }
                    #endif
                }
            }
            // Set origin to zero (match zeldovich.cpp line 572)
            for (int a = 0; a < narray; a++) {
                PRIM_SLICE(a, 0, 0)[0] = 0.0;
                PRIM_SLICE(a, 0, 0)[1] = 0.0;
            }
        }
        // Set imaginary parts to 0 at special points
        for (int a = 0; a < narray; a++) {
            PRIM_SLICE(a, 0, 0)[1] = 0.0;
            PRIM_SLICE(a, Nhalf, 0)[1] = 0.0;
            PRIM_SLICE(a, 0, Nhalf)[1] = 0.0;
            PRIM_SLICE(a, Nhalf, Nhalf)[1] = 0.0;
        }
    }
    pt_mirror.Stop(0);
    
    t_zloop_end = omp_get_wtime();
    
    // Verify Hermitian symmetry BEFORE 2D FFT
    // STAGE 7: Updated to check all arrays independently
    #if VERIFY_HERMITIAN_SYMMETRY
    for (int a = 0; a < narray; a++) {
        fftw_complex_t *prim_array = &PRIM_SLICE(a, 0, 0);
        fftw_complex_t *conj_array = (y_mirror != global_y) ? &CONJ_SLICE(a, 0, 0) : NULL;
        verify_initial_fourier_hermitian_symmetry(N, prim_array, global_y, y_mirror, conj_array, a);
    }
    #endif
    
    t_verify_end = omp_get_wtime();
    
    // 2D FFT: staged copy — plan_dft_2d on stage_2d, loop over narray
    for (int a = 0; a < narray; a++) {
        // plane a starts a * N * N elements after the base.
        fftw_complex_t *plane = primary_slices + (size_t)a * (size_t)N * (size_t)N;

        // copy to stage_2d
        pt_fft_copy_in.Start(0);
        copy_plane_chunked(stage_2d, plane, N);
        pt_fft_copy_in.Stop(0);

        // execute FFTW_EXECUTE_DFT on stage_2d
        pt_fft_execute.Start(0);
        FFTW_EXECUTE_DFT(plan_2d, stage_2d, stage_2d);
        pt_fft_execute.Stop(0);

        // copy back to primary_slices
        pt_fft_copy_out.Start(0);
        copy_plane_chunked(plane, stage_2d, N);
        pt_fft_copy_out.Stop(0);
    }

    if (y_mirror != global_y) {
        for (int a = 0; a < narray; a++) {
            // plane a starts a * N * N elements after the base.
            fftw_complex_t *plane = conjugate_slices + (size_t)a * (size_t)N * (size_t)N;
            pt_fft_copy_in.Start(0);
            copy_plane_chunked(stage_2d, plane, N);
            pt_fft_copy_in.Stop(0);
            pt_fft_execute.Start(0);
            FFTW_EXECUTE_DFT(plan_2d, stage_2d, stage_2d);
            pt_fft_execute.Stop(0);
            pt_fft_copy_out.Start(0);
            copy_plane_chunked(plane, stage_2d, N);
            pt_fft_copy_out.Stop(0);
        }
    }
    
    t_fft_end = omp_get_wtime();
    
    #if DEBUG_PRINTS
    // ========== DIAGNOSTIC: Print per-Y timing breakdown ==========
    // Print for first few Y-slices and occasionally thereafter to avoid flooding
    if (global_y <= 3 || (global_y % 64 == 0)) {
        fprintf(stderr, "[SLICE-TIMING] Rank=%d Y=%d/%d | setup=%.3fms zloop=%.3fms verify=%.3fms fft=%.3fms total=%.3fms\n",
                rank, global_y, y_mirror,
                (t_setup_end - t_func_start) * 1000.0,
                (t_zloop_end - t_setup_end) * 1000.0,
                (t_verify_end - t_zloop_end) * 1000.0,
                (t_fft_end - t_verify_end) * 1000.0,
                (t_fft_end - t_func_start) * 1000.0);
        fflush(stderr);
    }
    #endif
    
    // ========== Apply any remaining nskip at end of function ==========
    // See zeldovich.cpp: Pk.v2rng[y].advance(2 * nskip)
    // Ensures that exactly MAX_PPD * MAX_PPD complex numbers (2 * MAX_PPD * MAX_PPD real numbers)
    // were consumed/skipped for this y-row
    // Y=N/2 has no v2rng slot in serial zeldovich (plane zeroed, not drawn).
    if (nskip > 0 && global_y != Nhalf) {
        if (ps_handle != NULL && params_handle != NULL) {
            int64_t rng_index = global_y;
            zeldovich_ps_advance_rng(ps_handle, params_handle, rng_index, nskip);
            #if VERIFY_RNG_CALLS
            total_rng_skips += nskip;
            #endif
        }
        nskip = 0;  // Reset after applying
    }
    
    // // ========== RNG VERIFICATION: Verify total RNG calls match expected ==========

    // #if VERIFY_RNG_CALLS
    // {
    //     int64_t max_ppd_val = (int64_t)MAX_PPD;
    //     int64_t expected_total = 2LL * max_ppd_val * max_ppd_val;
    //     int64_t actual_total = 2LL * total_rng_calls + 2LL * total_rng_skips;
        
    //     if (actual_total != expected_total) {
    //         fprintf(stderr, 
    //                 "[RNG-VERIFY ERROR] Rank %d, Y=%d: Expected %ld random numbers (MAX_PPD=%ld), got %ld "
    //                 "(calls=%ld * 2 = %ld, skips=%ld * 2 = %ld)\n",
    //                 rank, global_y, (long)expected_total, (long)max_ppd_val, (long)actual_total,
    //                 (long)total_rng_calls, (long)(2LL * total_rng_calls),
    //                 (long)total_rng_skips, (long)(2LL * total_rng_skips));
    //         fflush(stderr);
    //         // Don't abort in production, but warn
    //         #ifdef DEBUG
    //         assert(actual_total == expected_total);
    //         #endif
    //     } else if (rank == 0 && global_y <= 2) {
    //         // Print verification for first few slices in debug mode
    //         fprintf(stderr,
    //                 "[RNG-VERIFY OK] Rank %d, Y=%d: Total=%ld (calls=%ld, skips=%ld)\n",
    //                 rank, global_y, (long)actual_total,
    //                 (long)total_rng_calls, (long)total_rng_skips);
    //         fflush(stderr);
    //     }
    // }
    // #endif
    
    // Clean up local macros
    #if LOAD_WHITE_NOISE_FROM_FILE
        free(loaded_noise_real);
        free(loaded_noise_imag);
    #endif

    #undef PRIM_SLICE
    #undef CONJ_SLICE
}

void print_zd_mpi_gen_timers(int rank) {
    double rng_s    = pt_rng_setup.Elapsed();
    double zloop_s  = pt_zloop.Elapsed();
    double mirror_s = pt_mirror.Elapsed();
    double fft_ci_s = pt_fft_copy_in.Elapsed();
    double fft_ex_s = pt_fft_execute.Elapsed();
    double fft_co_s = pt_fft_copy_out.Elapsed();
    double gen_s    = rng_s + zloop_s + mirror_s;
    double total    = gen_s + fft_ci_s + fft_ex_s + fft_co_s;
    if (rank == 0) {
        fprintf(stdout,
            "  [PTimerWall] Stage 1 sub-phase breakdown (accumulated over all Y-slices):\n"
            "    RNG setup (alloc+copy+adv): %10.6f s  (%5.1f%%)\n"
            "    Z-loop compute:             %10.6f s  (%5.1f%%)\n"
            "    Mirror (self-conjugate):     %10.6f s  (%5.1f%%)\n"
            "    Generation subtotal:         %10.6f s  (%5.1f%%)\n"
            "    2D copy-in (staged):         %10.6f s  (%5.1f%%)\n"
            "    2D FFT execute (dft_2d):     %10.6f s  (%5.1f%%)\n"
            "    2D copy-out (staged):        %10.6f s  (%5.1f%%)\n"
            "    Sum:                         %10.6f s\n",
            rng_s,    total > 0 ? 100.0 * rng_s    / total : 0.0,
            zloop_s,  total > 0 ? 100.0 * zloop_s  / total : 0.0,
            mirror_s, total > 0 ? 100.0 * mirror_s / total : 0.0,
            gen_s,    total > 0 ? 100.0 * gen_s    / total : 0.0,
            fft_ci_s, total > 0 ? 100.0 * fft_ci_s / total : 0.0,
            fft_ex_s, total > 0 ? 100.0 * fft_ex_s / total : 0.0,
            fft_co_s, total > 0 ? 100.0 * fft_co_s / total : 0.0,
            total);
        fflush(stdout);
    }
    pt_rng_setup.Clear();
    pt_zloop.Clear();
    pt_mirror.Clear();
    pt_fft_copy_in.Clear();
    pt_fft_execute.Clear();
    pt_fft_copy_out.Clear();
}
