// ====================================================================================
// C++ wrapper for zeldovich-PLT P(k) and Parameters classes
// ====================================================================================

#include "zeldovich_wrapper.h"
#include <power_spectrum.h>
#include <parameters.h>
#include <zeldovich.h>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" {

// --- Parameters ---

ParametersHandle zeldovich_params_create(const char* param_file) {
    try {
        ZeldovichParameters* params = new ZeldovichParameters(fs::path(param_file));
        return static_cast<ParametersHandle>(params);
    } catch (...) {
        return NULL;
    }
}

ParametersHandle zeldovich_params_create_from_buffer(
    const char* header_bytes,
    size_t header_len,
    const char* source_name
) {
    if (header_bytes == NULL || header_len == 0 || source_name == NULL) {
        return NULL;
    }
    try {
        ZeldovichParameters* params = new ZeldovichParameters(header_bytes, header_len, fs::path(source_name));
        return static_cast<ParametersHandle>(params);
    } catch (...) {
        return NULL;
    }
}

void zeldovich_params_destroy(ParametersHandle params) {
    if (params) {
        delete static_cast<ZeldovichParameters*>(params);
    }
}

// Pattern: Cast to Parameters object and attribute
double zeldovich_params_get_fundamental(ParametersHandle params) {
    if (!params) return 0.0;
    // Cast to Parameters object and return fundamental
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->fundamental;
}

double zeldovich_params_get_boxsize(ParametersHandle params) {
    if (!params) return 0.0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->boxsize;
}

double zeldovich_params_get_Pk_scale(ParametersHandle params) {
    if (!params) return 0.0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->Pk_scale;
}

int64_t zeldovich_params_get_ppd(ParametersHandle params) {
    if (!params) return 0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->ppd;
}

int zeldovich_params_get_cpd(ParametersHandle params) {
    if (!params) return 0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->cpd;
}

int zeldovich_params_get_NumZRanks(ParametersHandle params) {
    if (!params) return 0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->num_z_ranks;
}

int abacus_params_get_NumZRanks(ParametersHandle params) {
    if (!params) return 0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->abacus_num_z_ranks;
}

int zeldovich_params_get_seed(ParametersHandle params) {
    if (!params) return 0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->seed;
}

double zeldovich_params_get_Pk_powerlaw_index(ParametersHandle params) {
    if (!params) return 1000.0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->Pk_powerlaw_index;
}

const char* zeldovich_params_get_Pk_filename(ParametersHandle params) {
    if (!params) return NULL;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    if (p->Pk_filename.empty()) return NULL;
    return p->Pk_filename.c_str();
}

const char* zeldovich_params_get_Pk_primordial_filename(ParametersHandle params) {
    if (!params) return NULL;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    if (p->Pk_primordial_filename.empty()) return NULL;
    return p->Pk_primordial_filename.c_str();
}

const char* zeldovich_params_get_png_dir(ParametersHandle params) {
    if (!params) return NULL;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    if (p->png_dir.empty()) return NULL;
    return p->png_dir.c_str();
}

const char* zeldovich_params_get_white_noise_name(ParametersHandle params) {
    if (!params) return NULL;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    if (p->white_noise_name.empty()) return NULL;
    return p->white_noise_name.c_str();
}

double zeldovich_params_get_f_cluster(ParametersHandle params) {
    if (!params) return 0.0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->f_cluster;
}

double zeldovich_params_get_z_initial(ParametersHandle params) {
    if (!params) return 0.0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->z_initial;
}

double zeldovich_params_get_Omega_M(ParametersHandle params) {
    if (!params) return 0.0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->Omega_M;
}

int zeldovich_params_get_qPLT(ParametersHandle params) {
    if (!params) return 0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->qPLT;
}

const char* zeldovich_params_get_PLT_filename(ParametersHandle params) {
    if (!params) return NULL;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    if (p->PLT_filename.empty()) return NULL;
    return p->PLT_filename.c_str();
}

int zeldovich_params_get_qPLTrescale(ParametersHandle params) {
    if (!params) return 0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->qPLTrescale;
}

double zeldovich_params_get_PLT_target_z(ParametersHandle params) {
    if (!params) return 0.0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->PLT_target_z;
}

const char* zeldovich_params_get_ICFormat(ParametersHandle params) {
    if (!params) return NULL;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->ICFormat.c_str();
}

const char* zeldovich_params_get_local_wisdom_dir(ParametersHandle params) {
    if (!params) return NULL;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    if (p->local_wisdom_dir.empty()) return NULL;
    return p->local_wisdom_dir.c_str();
}

int zeldovich_params_get_qdensity(ParametersHandle params) {
    if (!params) return 0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->qdensity;
}

double zeldovich_params_get_k_cutoff(ParametersHandle params) {
    if (!params) return 1.0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->k_cutoff;
}

double zeldovich_params_get_f_NL(ParametersHandle params) {
    if (!params) return 0.0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->f_NL;
}

double zeldovich_params_get_f_NL_shape(ParametersHandle params, int i) {
    if (!params || i < 0 || i >= 4) return 0.0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->f_NL_shape[i];
}

double zeldovich_params_get_Pk_norm(ParametersHandle params) {
    if (!params) return 0.0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->Pk_norm;
}

void zeldovich_params_set_Pk_norm(ParametersHandle params, double val) {
    if (!params) return;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    p->Pk_norm = val;
}

int zeldovich_params_get_CornerModes(ParametersHandle params) {
    if (!params) return 0;
    ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
    return p->CornerModes;
}

// --- PowerSpectrum ---

static const int MAX_TRACKED_OBJECTS = 1024;
static PowerSpectrum* destroyed_objects[MAX_TRACKED_OBJECTS];
static int num_destroyed = 0;

PowerSpectrumHandle zeldovich_ps_create(int n, ParametersHandle params) {
    if (!params) return NULL;
    try {
        ZeldovichParameters* p = static_cast<ZeldovichParameters*>(params);
        PowerSpectrum* ps = new PowerSpectrum(n, *p);
        return static_cast<PowerSpectrumHandle>(ps);
    } catch (...) {
        return NULL;
    }
}

void zeldovich_ps_destroy(PowerSpectrumHandle ps) {
    if (!ps) return;
    PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
    for (int i = 0; i < num_destroyed; i++) {
        if (destroyed_objects[i] == p)
            return;
    }
    if (num_destroyed < MAX_TRACKED_OBJECTS)
        destroyed_objects[num_destroyed++] = p;
    try {
        delete p;
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] zeldovich_ps_destroy: %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "[ERROR] zeldovich_ps_destroy: unknown exception\n");
    }
}

int zeldovich_ps_init_powerlaw(PowerSpectrumHandle ps, double powerlaw_index, ParametersHandle params) {
    if (!ps || !params) return -1;
    try {
        PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
        ZeldovichParameters* par = static_cast<ZeldovichParameters*>(params);
        return p->InitFromPowerLaw(powerlaw_index, *par);
    } catch (...) {
        return -1;
    }
}

int zeldovich_ps_init_file(PowerSpectrumHandle ps, const char* filename, ParametersHandle params) {
    if (!ps || !params) return -1;
    try {
        PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
        ZeldovichParameters* par = static_cast<ZeldovichParameters*>(params);
        return p->InitFromFile(fs::path(filename), *par);
    } catch (...) {
        return -1;
    }
}

int zeldovich_ps_init_from_raw_pk(PowerSpectrumHandle ps, const double* k, const double* p,
    size_t n, ParametersHandle params) {
    if (!ps || !params || !k || !p || n == 0) return -1;
    try {
        PowerSpectrum* ps_ptr = static_cast<PowerSpectrum*>(ps);
        ZeldovichParameters* par = static_cast<ZeldovichParameters*>(params);
        return ps_ptr->InitFromRawPk(k, p, n, *par);
    } catch (...) {
        return -1;
    }
}

int zeldovich_ps_init_primordial_from_raw_pk(PowerSpectrumHandle ps, const double* k, const double* p,
    size_t n, ParametersHandle params) {
    if (!ps || !params || !k || !p || n == 0) return -1;
    try {
        PowerSpectrum* ps_ptr = static_cast<PowerSpectrum*>(ps);
        ZeldovichParameters* par = static_cast<ZeldovichParameters*>(params);
        return ps_ptr->InitPrimordialFromRawPk(k, p, n, *par);
    } catch (...) {
        return -1;
    }
}

double zeldovich_ps_power(PowerSpectrumHandle ps, double wavenumber) {
    if (!ps) return 0.0;
    PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
    return p->power(wavenumber);
}

double zeldovich_ps_one_rand(PowerSpectrumHandle ps, int64_t rng_index) {
    if (!ps) return 0.0;
    PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
    return p->one_rand<2>(rng_index);
}

//Charlie: adding this so we can calculate the primordial power spectrum. We end up not using this method because it is less accurate than using a primordial power spectrum uploaded directly from a CLASS file
double zeldovich_ps_primordial_power(PowerSpectrumHandle ps, double wavenumber) {
    if (!ps) return 0.0;
    PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
    return p->primordial_power(wavenumber);
}

void zeldovich_ps_cgauss(PowerSpectrumHandle ps, double wavenumber, int64_t rng_index, double* real, double* imag) { //Charlie:cgauss
    if (!ps || !real || !imag) return;
    PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
    Complx result = p->cgauss<2>(wavenumber, rng_index);
    *real = result.real();
    *imag = result.imag();
}

//Charlie: added for inputted IC from file
void zeldovich_ps_inputted_ic(PowerSpectrumHandle ps, double wavenumber, double real, double imag, double* Dreal, double* Dimag) { 
    if (!ps || !Dreal || !Dimag) return;
    PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
    Complx result = p->inputted_ic(wavenumber, real, imag);
    *Dreal = result.real();
    *Dimag = result.imag();
}

void zeldovich_ps_advance_rng(PowerSpectrumHandle ps, ParametersHandle params, int64_t rng_index, int64_t nskip) { //Charlie: what is this code?
    if (!ps || !params || nskip <= 0) return;
    PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
    ZeldovichParameters* param = static_cast<ZeldovichParameters*>(params);
    int64_t ppd_half = param->ppd / 2;
    if (rng_index >= 0 && rng_index < ppd_half && p->v2rng) {
        uint64_t advance_amount = (uint64_t)2 * (uint64_t)nskip;
        p->v2rng[rng_index].advance(advance_amount);
    }
}


// All threads read from the same v2rng[global_y] and write into their own buffers.
void zeldovich_ps_get_rng_copy(PowerSpectrumHandle ps, int64_t rng_index, void* out_rng) {
    if (!ps || !out_rng) return;
    PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
    pcg64 copy = p->get_rng_copy(rng_index);
    new (out_rng) pcg64(copy);
}

size_t zeldovich_ps_rng_buffer_size(void) {
    return sizeof(pcg64);
}

void zeldovich_ps_advance_rng_buffer(void* rng_buf, int64_t nskip) {
    if (!rng_buf || nskip <= 0) return;
    pcg64* rng = static_cast<pcg64*>(rng_buf);
    rng->advance((uint64_t)2 * (uint64_t)nskip);
}

// Match zeldovich one_rand<2>: returns (0,1]
static double one_rand_from_pcg64(pcg64* rng) {
    uint64_t r = (*rng)();
    if (r == UINT64_MAX) return 1.0;
    r += (uint64_t)1;
    return ldexp(static_cast<double>(r), -64);
}

void zeldovich_ps_cgauss_from_buffer(void* rng_buf, PowerSpectrumHandle ps, double wavenumber, double* real, double* imag) {
    if (!rng_buf || !ps || !real || !imag) return;
    pcg64* rng = static_cast<pcg64*>(rng_buf);
    PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
    double Pk = p->power(wavenumber);
    double R = one_rand_from_pcg64(rng);
    double theta = one_rand_from_pcg64(rng);
    if (p->fixed_power)
        R = sqrt(Pk);
    else
        R = sqrt(-Pk * log(R));
    theta = 2 * M_PI * theta;
    *real = R * cos(theta);
    *imag = R * sin(theta);
}

double zeldovich_ps_get_normalization(PowerSpectrumHandle ps) {
    if (!ps) return 0.0;
    PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
    return p->normalization;
}

double zeldovich_ps_get_powerlaw_index(PowerSpectrumHandle ps) {
    if (!ps) return 0.0;
    PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
    return p->powerlaw_index;
}

double zeldovich_ps_get_Pk_smooth2(PowerSpectrumHandle ps) {
    if (!ps) return 0.0;
    PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
    return p->Pk_smooth2;
}

int zeldovich_ps_get_fixed_power(PowerSpectrumHandle ps) {
    if (!ps) return 0;
    PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
    return p->fixed_power;
}

// // Charlie: added so a secondary PowerSpectrum (e.g. the primordial-dimensional
// // file) can be forced to carry the SAME normalization constant as the main
// // P(k) file, rather than the self-referential sigma8 rescaling that
// // Normalize() would otherwise compute from its own (different) shape.
// void zeldovich_ps_set_normalization(PowerSpectrumHandle ps, double normalization) {
//     if (!ps) return;
//     PowerSpectrum* p = static_cast<PowerSpectrum*>(ps);
//     p->normalization = normalization;
// }

} 

int zeldovich_pk_load_text_file_vectors(const char* filename, ParametersHandle params,
    std::vector<double>& k_out, std::vector<double>& p_out)
{
    if (!filename || !params) return -1;
    try {
        ZeldovichParameters* par = static_cast<ZeldovichParameters*>(params);
        return ReadPkTextFileIntoVectors(fs::path(filename), *par, k_out, p_out);
    } catch (...) {
        return -1;
    }
}

