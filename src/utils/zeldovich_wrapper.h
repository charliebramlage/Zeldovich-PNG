#ifndef ZELDOVICH_WRAPPER_H
#define ZELDOVICH_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer type for PowerSpectrum object
typedef void* PowerSpectrumHandle;

// Opaque pointer type for Parameters object
typedef void* ParametersHandle;

// ====================================================================================
// PARAMETERS INTERFACE
// ====================================================================================

// Create Parameters object from parameter file
// Returns NULL on error
ParametersHandle zeldovich_params_create(const char* param_file);

// Create Parameters object from in-memory header bytes
// Buffer must include the ParseHeader expected trailing "\0\0".
// Returns NULL on error.
ParametersHandle zeldovich_params_create_from_buffer(
    const char* header_bytes,
    size_t header_len,
    const char* source_name
);

// Destroy Parameters object
void zeldovich_params_destroy(ParametersHandle params);

// Get fundamental wavenumber (2pi/boxsize)
double zeldovich_params_get_fundamental(ParametersHandle params);

// Get boxsize
double zeldovich_params_get_boxsize(ParametersHandle params);

// Get Pk_scale
double zeldovich_params_get_Pk_scale(ParametersHandle params);

// Get ppd (grid size)
int64_t zeldovich_params_get_ppd(ParametersHandle params);

// Get cpd (coarse particle decomposition; number of slabs for output alignment)
int zeldovich_params_get_cpd(ParametersHandle params);

// Get user-specified number of ranks along z (ZD_NumZRanks)
int zeldovich_params_get_NumZRanks(ParametersHandle params);

// Get Abacus NumZRanks (no ZD_ prefix): z-dimension ranks in Abacus 2D decomposition
int abacus_params_get_NumZRanks(ParametersHandle params);

// Get seed
int zeldovich_params_get_seed(ParametersHandle params);

// Get Pk_powerlaw_index
double zeldovich_params_get_Pk_powerlaw_index(ParametersHandle params);

// Get Pk_filename (tabular P(k) input). NULL if empty — use power law in that case.
// Valid only while Parameters object exists.
const char* zeldovich_params_get_Pk_filename(ParametersHandle params);

// Get Pk_primordial_filename (ZD_Pk_primordial_filename in the .par file). Defaults
// to "class_pk_primordial_dimensional.dat" if not set. NULL only if the Parameters
// object itself is NULL. Valid only while Parameters object exists.
const char* zeldovich_params_get_Pk_primordial_filename(ParametersHandle params);

// Directory for Phi_gaussian_y.<y> and kernel_y.<y> (ZD_PNG_DIR). Default "png_slices".
// Valid only while the Parameters object exists.
const char* zeldovich_params_get_png_dir(ParametersHandle params);

// Get fundamental wavenumber (2pi/BoxSize)
double zeldovich_params_get_fundamental(ParametersHandle params);

// Get f_cluster (fraction of matter that is clustering)
double zeldovich_params_get_f_cluster(ParametersHandle params);

// Get z_initial (initial redshift)
double zeldovich_params_get_z_initial(ParametersHandle params);

// Get Omega_M (Omega_M at z=0; only used for f_NL)
double zeldovich_params_get_Omega_M(ParametersHandle params);

// Get qPLT (PLT flag: non-zero if using Particle Linear Theory modes)
int zeldovich_params_get_qPLT(ParametersHandle params);

// Get PLT_filename (file containing PLT eigenmodes)
// Returns: C string (caller should not free). Returns NULL if empty.
// Note: The string is valid only while Parameters object exists.
const char* zeldovich_params_get_PLT_filename(ParametersHandle params);

// Get qPLTrescale (rescaling flag: non-zero to rescale initial amplitudes)
int zeldovich_params_get_qPLTrescale(ParametersHandle params);

// Get PLT_target_z (target redshift for PLT rescaling)
double zeldovich_params_get_PLT_target_z(ParametersHandle params);

// Get ICFormat (output format string, e.g., "RV", "RVDoubleZel", etc.)
// Returns: C string (caller should not free). Returns NULL if empty.
// Note: The string is valid only while Parameters object exists.
const char* zeldovich_params_get_ICFormat(ParametersHandle params);

// Get per-rank local FFTW wisdom directory (e.g. /dev/shm/Abacus_wisdom)
// Returns: C string (caller should not free). Returns NULL if empty.
const char* zeldovich_params_get_local_wisdom_dir(ParametersHandle params);

// Get qdensity (density output mode)
// Returns: 0 = normal mode, 1 = normal + density file, 2 = density only (no displacements)
int zeldovich_params_get_qdensity(ParametersHandle params);

// Get k_cutoff (wavenumber cutoff factor)
// Returns: k_cutoff value (default 1.0, corresponds to k_Nyquist)
double zeldovich_params_get_k_cutoff(ParametersHandle params);

// Get f_NL (ZD_f_NL in the .par file; primordial non-Gaussianity amplitude)
// Returns: f_NL value (default 0.0, i.e. Gaussian)
double zeldovich_params_get_f_NL(ParametersHandle params);

// Get the f_NL shape flag i = 0..3 (ZD_f_NL_local, _equil, _orthoCMB, _orthoLSS).
// Nonzero means kernel_<shape>_y.{y} is loaded. Returns 0.0 if unset or i is out of range.
double zeldovich_params_get_f_NL_shape(ParametersHandle params, int i);

// Get/set Pk_norm (radius, Mpc/h, at which sigmaR is evaluated for sigma8-style
// normalization; if <= 0, that normalization branch is skipped entirely in
// Normalize()). The setter exists so a secondary PowerSpectrum can be loaded
// with this normalization temporarily disabled -- e.g. when its own shape makes
// the Romberg integration in sigmaR() expensive/slow to converge, and its
// normalization will be overwritten afterward anyway via
// zeldovich_ps_set_normalization(). Restore the original value after use.
double zeldovich_params_get_Pk_norm(ParametersHandle params);
void zeldovich_params_set_Pk_norm(ParametersHandle params, double val);

// Get CornerModes (corner mode handling flag)
// Returns: 0 = zero modes with k^2 >= k2_cutoff (default), non-zero = keep corner modes
int zeldovich_params_get_CornerModes(ParametersHandle params);

// ====================================================================================
// POWER SPECTRUM INTERFACE
// ====================================================================================

// Create PowerSpectrum object
PowerSpectrumHandle zeldovich_ps_create(int n, ParametersHandle params);

// Destroy PowerSpectrum object
void zeldovich_ps_destroy(PowerSpectrumHandle ps);

// Initialize from power law
int zeldovich_ps_init_powerlaw(PowerSpectrumHandle ps, double powerlaw_index, ParametersHandle params);

// Initialize from file
int zeldovich_ps_init_file(PowerSpectrumHandle ps, const char* filename, ParametersHandle params);

// Initialize from raw tabulated k,P - use after MPI bcast
int zeldovich_ps_init_from_raw_pk(PowerSpectrumHandle ps, const double* k, const double* p,
    size_t n, ParametersHandle params);

// Initialize the primordial spectrum from raw tabulated k,P: normalization is
// always exactly 1/BoxSize^3 (no Pk_norm/Pk_sigma/Pk_sigma_ratio, no Pk_smooth)
int zeldovich_ps_init_primordial_from_raw_pk(PowerSpectrumHandle ps, const double* k, const double* p,
    size_t n, ParametersHandle params);

// Evaluate power spectrum at given wavenumber
double zeldovich_ps_power(PowerSpectrumHandle ps, double wavenumber);

// Generate random number using zeldovich-PLT's v2rng
double zeldovich_ps_one_rand(PowerSpectrumHandle ps, int64_t rng_index);

//Charlie: added so we can calculate primordial power spectrum
double zeldovich_ps_primordial_power(PowerSpectrumHandle ps, double wavenumber);

//Charlie: added to calculate from a random number "file"
void zeldovich_ps_inputted_ic(PowerSpectrumHandle ps, double wavenumber, double real, double imag, double* Dreal, double* Dimag);

// Generate complex Gaussian random number with power spectrum variance
void zeldovich_ps_cgauss(PowerSpectrumHandle ps, double wavenumber, int64_t rng_index, double* real, double* imag);

// Advance RNG for given Y-slice index
void zeldovich_ps_advance_rng(PowerSpectrumHandle ps, ParametersHandle params, int64_t rng_index, int64_t nskip);

// Thread-local RNG support for parallel z-loop
// Get a copy of the RNG for Y-slice rng_index. Caller allocates out_rng (size from zeldovich_ps_rng_buffer_size).
void zeldovich_ps_get_rng_copy(PowerSpectrumHandle ps, int64_t rng_index, void* out_rng);
// Size in bytes for RNG buffer allocation
size_t zeldovich_ps_rng_buffer_size(void);
// Advance RNG in buffer by nskip (units: complex numbers = 2 random numbers each)
void zeldovich_ps_advance_rng_buffer(void* rng_buf, int64_t nskip);
// cgauss using RNG in buffer; needs ps for P(k) and fixed_power
void zeldovich_ps_cgauss_from_buffer(void* rng_buf, PowerSpectrumHandle ps, double wavenumber, double* real, double* imag);

// Get normalization
double zeldovich_ps_get_normalization(PowerSpectrumHandle ps);

// Get powerlaw_index
double zeldovich_ps_get_powerlaw_index(PowerSpectrumHandle ps);

// Get Pk_smooth2
double zeldovich_ps_get_Pk_smooth2(PowerSpectrumHandle ps);

// Get fixed_power flag
int zeldovich_ps_get_fixed_power(PowerSpectrumHandle ps);

// Force-set normalization (bypasses the sigma8/Pk_norm self-rescaling done in
// Normalize()). Use this to make a secondary PowerSpectrum (e.g. loaded from a
// different tabulated file) carry the SAME multiplicative normalization
// constant as another already-initialized PowerSpectrum -- e.g.
//   zeldovich_ps_set_normalization(ps_secondary, zeldovich_ps_get_normalization(ps_main));
// This matters whenever the main P(k) file has been sigma8-renormalized: that
// renormalization is self-referential (computed from that file's own shape),
// so a second file loaded independently would otherwise get a DIFFERENT,
// physically-inconsistent rescaling based on its own shape, which would
// corrupt any ratio computed between the two spectra (e.g. sqrt(P2/P1)).
void zeldovich_ps_set_normalization(PowerSpectrumHandle ps, double normalization);

#ifdef __cplusplus
}

#include <vector>

// Rank 0 only (MPI): read P(k) text file into vectors for broadcasting
// (Pk_scale applied: k *= Pk_scale, P /= Pk_scale^3)
int zeldovich_pk_load_text_file_vectors(const char* filename, ParametersHandle params,
    std::vector<double>& k_out, std::vector<double>& p_out);

#endif

#endif

