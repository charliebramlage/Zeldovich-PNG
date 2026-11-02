#include <stdint.h>
#include <stdio.h>

#include <vector>

#include "parameters.h"
#include "power_spectrum.h"

int ReadPkTextFileIntoVectors(const fs::path &filename, ZeldovichParameters &param,
    std::vector<double> &k_out, std::vector<double> &p_out)
{
    char line[200];
    FILE *fp;
    double k, P;
    fp = fopen(filename.c_str(), "r");
    if (fp == NULL) {
        fmt::print(stderr, "Power spectrum file \"{}\" not found.\n", filename);
        return -1;
    }
    k_out.clear();
    p_out.clear();
    while (fgets(line, 200, fp) != NULL) {
        if (line[0] == '#') continue;
        if (sscanf(line, "%lf %lf", &k, &P) < 2) continue;
        if (k < 0.0) continue;
        if (P < 0.0) continue;
        // Pk_scale converts the file's units to BoxSize units: k_box = Pk_scale * k_file.
        // P has units of length^3, so it converts by the inverse cube. Upstream only scaled k,
        // which ZD_Pk_sigma's sigma(R) matching hides, but the 1/BoxSize^3-only primordial
        // normalization and ZD_Pk_sigma_ratio use P's amplitude directly.
        // E.g. BoxSize in Mpc, file in h/Mpc and (Mpc/h)^3, Pk_scale = h: P_Mpc^3 = P / h^3.
        k *= param.Pk_scale;
        P /= param.Pk_scale * param.Pk_scale * param.Pk_scale;
        k_out.push_back(k);
        p_out.push_back(P);
    }
    fclose(fp);
    return 0;
}

PowerSpectrum::PowerSpectrum(int n, ZeldovichParameters &param) : SplineFunction(n) {
    is_powerlaw    = 0;
    powerlaw_index = 1000;
    kmin           = std::numeric_limits<double>::max();
    kmax           = std::numeric_limits<double>::min();

    // Set up multiple RNGs to support parallelism and over-/down-sampling
    // seed the rng. a seed of zero uses the current time
    unsigned long int longseed = param.seed;
    block       = 0;
    v2rng_count = 0;
    n_s         = param.n_s;

#ifdef HAVE_GSL
    v1rng = NULL;
    v2rng = NULL;

    if (param.version == 1) {
        block = param.ppd / param.numblock;
        v1rng = new gsl_rng *[block];

        for (int i = 0; i < block; i++) {
            v1rng[i] = gsl_rng_alloc(gsl_rng_mt19937);
            gsl_rng_set(v1rng[i], longseed + i);
        }
    } else
#endif
    {
        // We'll make ppd/2 independent y-planes
        // But we do so by fast-forwarding the base RNG,
        // so logically this is just a single output stream from the RNG
        v2rng_count = param.ppd / 2;
        v2rng       = new pcg64[v2rng_count];
        v2rng[0] = pcg64(longseed);
        for (int i = 1; i < v2rng_count; i++) {
            v2rng[i] = v2rng[i - 1];
            // Each plane is ppd^2 complexes
            v2rng[i].advance(2 * MAX_PPD * MAX_PPD);
        }
    }
}

PowerSpectrum::~PowerSpectrum() {
#ifdef HAVE_GSL
    if (v1rng != NULL) {
        for (int64_t i = 0; i < block; i++) gsl_rng_free(v1rng[i]);
        delete[] v1rng;
    } else
#endif
    {
        delete[] v2rng;
    }
}

double PowerSpectrum::sigmaR_integrand(double k) {
    double x = k * Rnorm;
    double w;
    if (x <= 1e-3)
        w = 1 - x * x / 10.0;
    else
        w = 3.0 * (sin(x) - x * cos(x)) / x / x / x;
    return 0.5 / M_PI / M_PI * k * k * w * w * this->power(k);
}

double PowerSpectrum::sigmaR(double R) {
    if (!is_powerlaw) {
        double target_prec = 1e-6;  // 1e-12 was causing divergence in certain cases
        double precision   = 1.0;
        Rnorm              = R;
        double retval      = sqrt(
           Romberg(&PowerSpectrum::sigmaR_integrand, 0, 10.0, target_prec, &precision)
        );
        if (precision > target_prec) {
            fmt::print(
               stderr,
               "Error: actual Romberg integration precision ({:g}) is greater than the target precision ({:g}); halting.\n",
               precision,
               target_prec
            );
            exit(1);
        }
        return retval;
    } else {
        // Use the analytic power law solution
        // See:
        // http://nbviewer.jupyter.org/gist/lgarrison/7e41ee280c57554e256b834ac5c3f753
        double n      = powerlaw_index;
        double retval = 9 * pow(R, -n - 3) / (2 * M_PI * sqrt(M_PI))
                        * tgamma((3 + n) / 2.)
                        / (tgamma((2 - n) / 2.) * (n - 3) * (n - 1));
        retval = sqrt(retval * normalization);
        return retval;
    }
}
// Do Romberg integration with up to MAXITER bisections.
// Give a precision 'prec'.  Also return the estimated precision in 'obtprec'.

#define MAXITER 32
double PowerSpectrum::Romberg(
   double (PowerSpectrum::*func)(double),
   double a,
   double b,
   double prec,
   double *obtprec
) {
    int jj;
    double h, s, fourtokm1, TT[MAXITER + 1][MAXITER + 1];

    h        = 0.5 * (b - a);
    TT[0][1] = h * ((this->*func)(a) + (this->*func)(b));
    jj       = 0;
    do {
        jj++;
        s = 0;
#pragma omp parallel for reduction(+ : s)
        for (uint64_t k = 1; k <= (1ULL << (jj - 1)); k++)
            s += (this->*func)(a + (2 * k - 1) * h);
        TT[jj][1] = 0.5 * TT[jj - 1][1] + h * s;
        fourtokm1 = 1;
        for (int k = 2; k <= jj; k++) {
            fourtokm1 *= 4;
            TT[jj][k] =
               TT[jj][k - 1] + (TT[jj][k - 1] - TT[jj - 1][k - 1]) / (fourtokm1 - 1);
        }
        h *= 0.5;
        // fmt::print(stderr,"TT[{:d}] = {:g}\n", jj, TT[jj][jj]);
        if (jj > 1 && fabs(TT[jj][jj] - TT[jj - 1][jj - 1]) < prec * fabs(TT[jj][jj]))
            break;
    } while (jj < MAXITER);

    *obtprec = (TT[jj][jj] - TT[jj - 1][jj - 1]) / TT[jj][jj];
    return TT[jj][jj];
}

int PowerSpectrum::LoadRawPk(const double *k_arr, const double *p_arr, size_t n) {
    if (n == 0 || k_arr == NULL || p_arr == NULL) return -1;
    for (size_t i = 0; i < n; i++) {
        double k = k_arr[i];
        double P = p_arr[i];
        if (k > 0.0) {
            this->load(log(k), log(P));
            kmin = std::min(k, kmin);
        } else {
            this->load(-1e3, log(P));
        }
        kmax = std::max(k, kmax);
    }
    this->spline();
    return 0;
}

int PowerSpectrum::InitFromRawPk(const double *k_arr, const double *p_arr, size_t n, ZeldovichParameters &param) {
    if (LoadRawPk(k_arr, p_arr, n) != 0) return -1;
    Normalize(param);
    return 0;
}

int PowerSpectrum::InitPrimordialFromRawPk(const double *k_arr, const double *p_arr, size_t n, ZeldovichParameters &param) {
    if (LoadRawPk(k_arr, p_arr, n) != 0) return -1;
    NormalizeBoxOnly(param);
    return 0;
}

int PowerSpectrum::InitFromFile(const fs::path &filename, ZeldovichParameters &param) {
    fmt::print(stderr, "Loading power spectrum from file \"{}\"\n", filename);
    std::vector<double> ks, Ps;
    if (ReadPkTextFileIntoVectors(filename, param, ks, Ps) != 0) {
        return -1;
    }
    return InitFromRawPk(ks.data(), Ps.data(), ks.size(), param);
}

int PowerSpectrum::InitFromPowerLaw(double _powerlaw_index, ZeldovichParameters &param) {
    assert(_powerlaw_index != 1000);
    powerlaw_index = _powerlaw_index;
    is_powerlaw    = 1;
    fmt::print(
       stderr, "Initializing power spectrum with power law index {:g}\n", powerlaw_index
    );
    kmin = 1e-4;  // Arbitrary; used by f_NL

    Normalize(param);
    return 0;
}

void PowerSpectrum::Normalize(ZeldovichParameters &param) {
    Pk_smooth2    = 0.0;
    normalization = 1.0;

    // Might still have to normalize things!
    if (param.Pk_norm > 0.0) {  // Do a normalization
        fmt::print(
           stderr, "Input sigma({:f}) = {:.6g}\n", param.Pk_norm, sigmaR(param.Pk_norm)
        );

        if (param.Pk_sigma > 0) {
            normalization = param.Pk_sigma / sigmaR(param.Pk_norm);
            normalization *= normalization;
        } else if (param.Pk_sigma_ratio > 0) {
            normalization = param.Pk_sigma_ratio * param.Pk_sigma_ratio;
        } else {
            assert(
               param.Pk_sigma > 0 || param.Pk_sigma_ratio > 0
            );  // Illegal state! We checked this in the params
        }

        fmt::print(
           stderr, "Final sigma({:f}) = {:.6g}\n", param.Pk_norm, sigmaR(param.Pk_norm)
        );
    }
    // Might need to normalize to the box volume.  This is appropriate
    // if the iFFT is like FFTW, i.e., not dividing by N.
    // This is because integrals over k bring in a phase space volume
    // per cell of (2*pi/L)^3
    normalization /= param.boxsize * param.boxsize * param.boxsize;
    Pk_smooth2 = param.Pk_smooth * param.Pk_smooth;

    fixed_power = param.qPk_fix_to_mean;
    if (fixed_power) fmt::print(stderr, "Fixing density mode amplitudes to sqrt(P(k))\n");

    primordial_norm = 1.;
    primordial_norm = this->power(this->kmin) / this->primordial_power(this->kmin);
}

void PowerSpectrum::NormalizeBoxOnly(ZeldovichParameters &param) {
    // Only the FFTW box-volume factor from Normalize(): the spectrum keeps the
    // amplitude of its input file. No sigma(R) matching, no Pk_smooth, no fixed power.
    normalization   = 1.0 / (param.boxsize * param.boxsize * param.boxsize);
    Pk_smooth2      = 0.0;
    fixed_power     = 0;
    primordial_norm = 1.;  // only used by primordial_power(), not by this spectrum's power()
}

double PowerSpectrum::power(double wavenumber) {
    // Evaluate the power spectrum and apply a smoothing.
    // Probably we should force P(k=0) to be 0.
    // The smoothing should be exp(-k^2 sig^2/2) for the density,
    // which is exp(-k^2 sig^2) for the power

    if (wavenumber <= 0.0) return 0.0;

    if (is_powerlaw) {
        // power law is raw k^n, times normalization and smoothing
        return std::pow(wavenumber, powerlaw_index)
               * exp(-wavenumber * wavenumber * this->Pk_smooth2) * normalization;
    } else {
        static bool already_warned = false;
        if (wavenumber > kmax && !already_warned) {
            fmt::print(
               stderr,
               R"(
*** WARNING: power spectrum spline interpolation was requested
past the maximum k ({:f}) that was provided in the input power
spectrum file.  The extrapolation should be well-behaved, but
make sure that this was expected.  Provide a power spectrum
that goes to at least k=10 (or higher if your k_Nyquist demands
it) to get rid of this warning.

)",
               kmax
            );
            already_warned = true;
        }
        return exp(
                  this->val(log(wavenumber))
                  - wavenumber * wavenumber * this->Pk_smooth2
               )

               * normalization;
    }
}

double PowerSpectrum::primordial_power(double wavenumber) {
    if (wavenumber <= 0.0) return 0.0;
    return primordial_norm * exp(log(wavenumber) * n_s);
}

double PowerSpectrum::infer_Tk(double wavenumber) {
    // We call this "infer_Tk" to make it clear that we're inferring the value
    // by assuming T(k) = 1 on large scales, rather than reading a file with
    // the transfer function.
    if (wavenumber <= 0.0) return 1.0;
    return sqrt(power(wavenumber) / primordial_power(wavenumber));
}

#ifdef HAVE_GSL
// ZD_Version 1
template <>
double PowerSpectrum::one_rand<1>(int64_t i) {
    return gsl_rng_uniform(v1rng[i]);
}
#endif

// ZD_Version 2
// Returns a random double in (0,1]
template <>
double PowerSpectrum::one_rand<2>(int64_t i) {

    uint64_t r = v2rng[i]();

    // Can't return 0!  That will immediately break the log in Box-Muller
    // But 1.0 is a valid value, so we can just add 1 to everything
    // thus shifting the domain from [0,1) to (0,1]
    // But first check if adding 1 would overflow
    if (r == UINT64_MAX) return 1.;

    // Is (0,1] the correct range, or (0,1)?
    // Consider cos(2*pi*r), where we want the full period to be sampled.
    // If we don't return 0, then we should return 1, so the "phase origin" can be
    // sampled. That's a simplification, but if we're wrong it's only by 1 part in 2**64
    r += (uint64_t) 1;

    // Turn the uint64 into a double
    // Converting to a double and dividing by 2^64 is not the best way
    // to use the full range of double values between 0 and 1,
    // but probably not at a level we care about.
    // See http://mumble.net/~campbell/tmp/random_real.c
    // or http://allendowney.com/research/rand/downey07randfloat.pdf
    return ldexp(r, -64);
}

#ifdef HAVE_GSL
template <>
Complx PowerSpectrum::cgauss<1>(double wavenumber, int64_t rng) {
    // Return a gaussian complex deviate scaled to the sqrt of the power
    // Box-Muller, adapted from Numerical Recipes
    // If fixed_power is set, the complex deviate always has amplitude sqrt(P(k))

    double Pk = this->power(wavenumber);
    double phase1, phase2, r2;
    // fmt::print(stderr,"P({:f}) = {:g}\n",wavenumber,Pk);
    do {
        phase1 = one_rand<1>(rng) * 2.0 - 1.0;
        phase2 = one_rand<1>(rng) * 2.0 - 1.0;
        r2     = phase1 * phase1 + phase2 * phase2;
    } while (!(r2 < 1.0 && r2 > 0.0));
    if (fixed_power) {
        r2 = sqrt(Pk / r2);
    } else {
        r2 = sqrt(-Pk * log(r2) / r2);  // Drop the factor of 2, so these Gaussians
                                        // have variance of 1/2.
    }
    // fmt::print(stderr,"cgauss: {:f} {:f}\n", phase1*r2, phase2*r2);
    return Complx(phase1 * r2, phase2 * r2);
}
#endif

// ZD_Version 2 must use this "deterministic" version of Box-Muller,
// which is guaranteed to make exactly 2 RNG calls (unlike the rejection
// sampling method in Version 1).  In theory, the trig calls make this way
// slower, but in practice the FFTs, IO, and memory movement dominate the runtime
template <>
Complx PowerSpectrum::cgauss<2>(double wavenumber, int64_t rng) {
    // Return a gaussian complex deviate scaled to the sqrt of the power
    // Box-Muller, adapted from Numerical Recipes
    // If fixed_power is set, the complex deviate always has amplitude sqrt(P(k))

    double Pk    = this->power(wavenumber);
    double R     = one_rand<2>(rng);
    double theta = one_rand<2>(rng);

    // Standard Box-Muller without the factor of 2
    if (!fixed_power)
        R = sqrt(-Pk * log(R));
    else
        R = sqrt(Pk);
    theta = 2 * M_PI * theta;

    double g1 = R * cos(theta);
    double g2 = R * sin(theta);

    return Complx(g1, g2);
}

//Charlie:adding this
Complx PowerSpectrum::inputted_ic(double wavenumber, double real, double imag) {

    double Pk    = this->power(wavenumber);
    double sqrtPk     = sqrt(Pk);

    double g1 = sqrtPk * real;
    double g2 = sqrtPk * imag;

    return Complx(g1, g2);
}


pcg64 PowerSpectrum::get_rng_copy(int64_t rng_index) const {
    // Return a copy of the RNG for thread-local use (hermitian OpenMP z-loop).
    // ZD_Version 2: index is global Y in [0, ppd/2); v2rng has v2rng_count entries.
    // Do NOT use `block` (ppd/numblock) here — that is the v1 slab height only.
    if (v2rng && rng_index >= 0 && rng_index < v2rng_count) {
        return v2rng[rng_index];  // PCG64 copy constructor
    }
    if (v2rng) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr,
                    "[RNG-WARNING] get_rng_copy: Y=%lld out of range [0, %d) "
                    "(v2rng_count=ppd/2). "
                    "Returning pcg64(0).\n",
                    (long long)rng_index, v2rng_count);
            warned = 1;
        }
    }
    return pcg64(0);
}
