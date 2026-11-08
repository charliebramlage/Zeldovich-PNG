// ====================================================================================
// PNG_Combine: white-noise files from Phi_gaussian and the f_NL kernels, without the
// full Zeldovich_MPI machinery. A streaming replacement for the LOAD_D_FROM_FILE pass.
// ====================================================================================
// For every primary y-plane (y = 0 .. N/2) and every mode (x, z) of that plane:
//
//     out = s * phi_G + f_NL * s * K_shape,   s = (5/3) / sqrt(P_prim(|k|)),  s = 0 on masked modes
//
// written to {out_dir}/phi_<shape>_fNL<f>_divided_y.<y>. This is the same arithmetic, mask and
// primordial spline as the load pass in generation/ZD_MPI_generation.c (see the comments marked
// "same as the load pass"), so the files agree with it bit for bit (up to 1 float32 ulp where the
// compiler contracts a multiply-add differently). Planes are independent: no FFTs, no exchange.
//
// Usage:
//   mpiexec -n R PNG_Combine param.par --set local:10,-10,30,-30 --set equil:150,-150 [--set ...]
//                            [--out-dir DIR] [--overwrite]
// Inputs ({ZD_PNG_DIR}): Phi_gaussian_y.<y> and kernel_<shape>_y.<y> for every shape in --set,
// float32 or float64. Outputs: float32, or float64 with ZD_PNG_FileFloat64 = 1. Output directory
// defaults to ZD_PNG_DIR. Existing output files are never overwritten unless --overwrite is given:
// every rank checks its planes first and the run aborts before writing anything.
// ====================================================================================

#include <mpi.h>
#include <omp.h>

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

#include "ParseHeader.hh"
#include "png_slice_io.h"
#include "zeldovich_wrapper.h"

namespace fs = std::filesystem;

static const char *kShapeNames[4] = {"local", "equil", "orthoCMB", "orthoLSS"};

#define FAIL(...) do { fprintf(stderr, "[PNG_Combine] ERROR: " __VA_ARGS__); fflush(stderr); \
                       MPI_Abort(MPI_COMM_WORLD, 1); } while (0)

struct OutputSet {
    int shape;      // index into kShapeNames
    double f_NL;
    std::string name;   // phi_<shape>_fNL<f>_divided
};

static int shape_index(const std::string &s)
{
    for (int i = 0; i < 4; i++) if (s == kShapeNames[i]) return i;
    return -1;
}

// Same as the driver: rank 0 reads the parameter header, every rank gets the bytes.
static ParametersHandle load_params(const char *param_file, int rank)
{
    std::vector<char> bytes;
    uint64_t len = 0;
    if (rank == 0) {
        HeaderStream hs{fs::path(param_file)};
        hs.ReadHeader();
        if (hs.buffer != NULL && hs.bufferlength >= 2 && hs.bufferlength <= (size_t)INT_MAX) {
            bytes.assign(hs.buffer, hs.buffer + hs.bufferlength);
            len = hs.bufferlength;
        }
    }
    MPI_Bcast(&len, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (len < 2) FAIL("cannot read parameter file %s\n", param_file);
    if (rank != 0) bytes.resize(len);
    MPI_Bcast(bytes.data(), (int)len, MPI_CHAR, 0, MPI_COMM_WORLD);
    ParametersHandle p = zeldovich_params_create_from_buffer(bytes.data(), bytes.size(), param_file);
    if (p == NULL) FAIL("cannot parse parameter file %s\n", param_file);
    return p;
}

// Same as the driver (zeldovich_mpi_driver.cpp, "Primordial power spectrum"): rank 0 reads the
// file, every rank builds the spline from the broadcast arrays with the box-only normalization.
static PowerSpectrumHandle load_primordial(ParametersHandle params, int rank)
{
    const char *fn = zeldovich_params_get_Pk_primordial_filename(params);
    std::string path = (fn != NULL && fn[0] != '\0') ? fn : "class_pk_primordial_dimensional.dat";
    std::vector<double> k, P;
    uint64_t n = 0;
    if (rank == 0) {
        if (zeldovich_pk_load_text_file_vectors(path.c_str(), params, k, P) == 0 && !k.empty()) n = k.size();
    }
    MPI_Bcast(&n, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (n == 0) FAIL("cannot read the primordial power spectrum %s (ZD_Pk_primordial_filename)\n", path.c_str());
    if (rank != 0) { k.resize(n); P.resize(n); }
    MPI_Bcast(k.data(), (int)n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(P.data(), (int)n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    PowerSpectrumHandle ps = zeldovich_ps_create((int)n, params);
    if (ps == NULL || zeldovich_ps_init_primordial_from_raw_pk(ps, k.data(), P.data(), (size_t)n, params) != 0) {
        FAIL("cannot build the primordial power spectrum from %s\n", path.c_str());
    }
    if (rank == 0) printf("  primordial P(k): %s (%llu points)\n", path.c_str(), (unsigned long long)n);
    return ps;
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const double t_start = MPI_Wtime();

    // ---------------- arguments ----------------
    const char *param_file = NULL;
    std::string out_dir_arg;
    bool overwrite = false;
    std::vector<OutputSet> sets;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--overwrite") {
            overwrite = true;
        } else if (a == "--out-dir" && i + 1 < argc) {
            out_dir_arg = argv[++i];
        } else if (a == "--set" && i + 1 < argc) {
            std::string spec = argv[++i];
            size_t colon = spec.find(':');
            int sh = (colon == std::string::npos) ? -1 : shape_index(spec.substr(0, colon));
            if (sh < 0) FAIL("--set %s: expected <local|equil|orthoCMB|orthoLSS>:<f_NL>[,<f_NL>...]\n", spec.c_str());
            std::string list = spec.substr(colon + 1);
            size_t pos = 0;
            while (pos <= list.size()) {
                size_t comma = list.find(',', pos);
                std::string tok = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                char *end = NULL;
                errno = 0;
                double f = strtod(tok.c_str(), &end);
                if (tok.empty() || *end != '\0' || errno != 0 || !std::isfinite(f)) {
                    FAIL("--set %s: '%s' is not a number\n", spec.c_str(), tok.c_str());
                }
                char fstr[64];
                png_format_fnl(fstr, sizeof(fstr), f);
                std::string name = std::string("phi_") + kShapeNames[sh] + "_fNL" + fstr + "_divided";
                bool dup = false;
                for (auto &o : sets) dup |= (o.name == name);
                if (!dup) sets.push_back({sh, f, name});
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } else if (a[0] != '-' && param_file == NULL) {
            param_file = argv[i];
        } else {
            FAIL("unknown argument %s\n", a.c_str());
        }
    }
    if (param_file == NULL) FAIL("usage: PNG_Combine param.par --set <shape>:<f_NL>[,...] [--set ...] [--out-dir DIR] [--overwrite]\n");
    if (sets.empty()) FAIL("no --set given: nothing to write\n");

    // ---------------- parameters (same keys as the load pass) ----------------
    ParametersHandle params = load_params(param_file, rank);
    const int N = (int)zeldovich_params_get_ppd(params);
    const int Nhalf = N / 2;
    const double fundamental = zeldovich_params_get_fundamental(params);
    const double k_cutoff = zeldovich_params_get_k_cutoff(params);
    const int CornerModes = zeldovich_params_get_CornerModes(params);
    const double k2_cutoff = ((double)Nhalf * (double)Nhalf) / (k_cutoff * k_cutoff);   // same as the load pass
    const int out_bytes = zeldovich_params_get_png_file_bytes(params);
    const char *png_dir_p = zeldovich_params_get_png_dir(params);
    const std::string in_dir = (png_dir_p != NULL && png_dir_p[0] != '\0') ? png_dir_p : "png_slices";
    const std::string out_dir = out_dir_arg.empty() ? in_dir : out_dir_arg;
    if (N <= 0 || N % 2 != 0) FAIL("N = %d from the parameter file must be positive and even\n", N);
    PowerSpectrumHandle ps_prim = load_primordial(params, rank);

    bool shape_used[4] = {false, false, false, false};
    for (auto &o : sets) shape_used[o.shape] = true;
    const int nplanes = Nhalf + 1;
    const size_t nvals = (size_t)N * (size_t)N;

    if (rank == 0) {
        printf("========================================================================\n");
        printf("PNG_Combine: N=%d  BoxSize=%g  ranks=%d  OMP threads/rank=%d\n", N,
               zeldovich_params_get_boxsize(params), nranks, omp_get_max_threads());
        printf("  k_cutoff=%g (k2_cutoff=%.1f)  CornerModes=%d  output type %s\n", k_cutoff, k2_cutoff,
               CornerModes, png_slice_type_name(out_bytes));
        printf("  in:  %s\n  out: %s%s\n", in_dir.c_str(), out_dir.c_str(), overwrite ? "  (--overwrite)" : "");
        for (auto &o : sets) printf("  set: %s  (shape %s, f_NL = %.17g)\n", o.name.c_str(), kShapeNames[o.shape], o.f_NL);
        printf("========================================================================\n");
        fflush(stdout);
        std::error_code ec;
        fs::create_directories(out_dir, ec);
        if (ec) FAIL("cannot create %s: %s\n", out_dir.c_str(), ec.message().c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ---------------- planes of this rank; check inputs and outputs before writing ----------------
    std::vector<int> my_planes;
    for (int y = rank; y < nplanes; y += nranks) my_planes.push_back(y);
    auto path_of = [](const std::string &dir, const std::string &stem, int y) {
        return dir + "/" + stem + "_y." + std::to_string(y);
    };
    long long bad_inputs = 0, existing = 0;
    std::string first_bad, first_existing;
    for (int y : my_planes) {
        std::vector<std::string> ins = {path_of(in_dir, "Phi_gaussian", y)};
        for (int s = 0; s < 4; s++) if (shape_used[s]) ins.push_back(path_of(in_dir, std::string("kernel_") + kShapeNames[s], y));
        for (auto &p : ins) {
            int fmt = png_slice_format(p.c_str(), nvals);
            if (fmt != PNG_SLICE_F32 && fmt != PNG_SLICE_F64) { if (!bad_inputs++) first_bad = p; }
        }
        for (auto &o : sets) {
            std::string p = path_of(out_dir, o.name, y);
            if (access(p.c_str(), F_OK) == 0) { if (!existing++) first_existing = p; }
        }
    }
    long long tot_bad = 0, tot_existing = 0;
    MPI_Allreduce(&bad_inputs, &tot_bad, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&existing, &tot_existing, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (tot_bad > 0) {
        if (bad_inputs) fprintf(stderr, "[PNG_Combine] rank %d: missing or wrong-size input, e.g. %s\n", rank, first_bad.c_str());
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) FAIL("%lld input files missing or not N*N complex float32/float64 (N=%d)\n", tot_bad, N);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (tot_existing > 0 && !overwrite) {
        if (existing) fprintf(stderr, "[PNG_Combine] rank %d: output exists, e.g. %s\n", rank, first_existing.c_str());
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) FAIL("%lld output files already exist; nothing was written (use --overwrite to replace them)\n", tot_existing);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ---------------- combine ----------------
    std::vector<double> phi(2 * nvals), s(nvals), phiGs(2 * nvals), K(2 * nvals), out(2 * nvals);
    double t_read = 0, t_comp = 0, t_write = 0;
    long long bytes_read = 0, bytes_written = 0;
    int done = 0;
    for (int y : my_planes) {
        double t0 = MPI_Wtime();
        std::string pphi = path_of(in_dir, "Phi_gaussian", y);
        int err = png_slice_read_double(pphi.c_str(), nvals, phi.data());
        if (err != 0) FAIL("rank %d: cannot read %s (error %d)\n", rank, pphi.c_str(), err);
        bytes_read += (long long)nvals * 2 * png_slice_format(pphi.c_str(), nvals);
        double t1 = MPI_Wtime();
        t_read += t1 - t0;

        // s and phi_G * s for this plane: same as the load pass (ZD_MPI_generation.c, z-loop)
        const int ky = (y > Nhalf) ? y - N : y;
        const int abs_ky = (ky < 0) ? -ky : ky;
        #pragma omp parallel for schedule(static)
        for (int z = 0; z < N; z++) {
            const int kz = (z > Nhalf) ? z - N : z;
            const int abs_kz = (kz < 0) ? -kz : kz;
            for (int x = 0; x < N; x++) {
                const int kx = (x > Nhalf) ? x - N : x;
                const int abs_kx = (kx < 0) ? -kx : kx;
                const int k2_int = kx * kx + ky * ky + kz * kz;
                const double k2 = (double)k2_int;
                const int is_nyquist = (abs_kx == Nhalf || abs_ky == Nhalf || abs_kz == Nhalf);
                const int zeroed_mode = (k2 == 0.0) || is_nyquist || (!CornerModes && (double)k2_int >= k2_cutoff);
                double sv = 0.0;
                if (!zeroed_mode) {
                    double k2_phys = k2 * fundamental * fundamental;
                    double kmag = sqrt(k2_phys);
                    double P_primordial = zeldovich_ps_power(ps_prim, kmag);
                    if (P_primordial > 0.0) sv = (5.0 / 3.0) / sqrt(P_primordial);
                }
                const size_t i = (size_t)z * (size_t)N + (size_t)x;
                s[i] = sv;
                phiGs[2 * i]     = phi[2 * i] * sv;
                phiGs[2 * i + 1] = phi[2 * i + 1] * sv;
            }
        }
        double t2 = MPI_Wtime();
        t_comp += t2 - t1;

        for (int sh = 0; sh < 4; sh++) {
            if (!shape_used[sh]) continue;
            double t3 = MPI_Wtime();
            std::string pk = path_of(in_dir, std::string("kernel_") + kShapeNames[sh], y);
            err = png_slice_read_double(pk.c_str(), nvals, K.data());
            if (err != 0) FAIL("rank %d: cannot read %s (error %d)\n", rank, pk.c_str(), err);
            bytes_read += (long long)nvals * 2 * png_slice_format(pk.c_str(), nvals);
            double t4 = MPI_Wtime();
            t_read += t4 - t3;
            for (auto &o : sets) {
                if (o.shape != sh) continue;
                double t5 = MPI_Wtime();
                const double f_NL = o.f_NL;
                // same as the load pass: divided = phiG_scaled + (f_NL * s) * kernel
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < nvals; i++) {
                    double fs = f_NL * s[i];
                    out[2 * i]     = phiGs[2 * i]     + fs * K[2 * i];
                    out[2 * i + 1] = phiGs[2 * i + 1] + fs * K[2 * i + 1];
                }
                double t6 = MPI_Wtime();
                t_comp += t6 - t5;
                std::string po = path_of(out_dir, o.name, y);
                if (!overwrite && access(po.c_str(), F_OK) == 0) FAIL("rank %d: %s appeared during the run\n", rank, po.c_str());
                err = png_slice_write_double(po.c_str(), nvals, out.data(), out_bytes);
                if (err != 0) FAIL("rank %d: cannot write %s (error %d)\n", rank, po.c_str(), err);
                bytes_written += (long long)nvals * 2 * out_bytes;
                t_write += MPI_Wtime() - t6;
            }
        }
        done++;
        if (rank == 0 && (done == 1 || done % (((int)my_planes.size() + 9) / 10 > 0 ? ((int)my_planes.size() + 9) / 10 : 1) == 0)) {
            printf("[PNG_Combine] rank 0: %d/%zu planes, %.1f s\n", done, my_planes.size(), MPI_Wtime() - t_start);
            fflush(stdout);
        }
    }

    // ---------------- summary ----------------
    double tmax[3] = {t_read, t_comp, t_write}, tm[3];
    MPI_Reduce(tmax, tm, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long long b[2] = {bytes_read, bytes_written}, bt[2];
    MPI_Reduce(b, bt, 2, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_total = MPI_Wtime() - t_start;
    if (rank == 0) {
        printf("[PNG_Combine] %zu set(s) x %d planes written; total %.1f s (max over ranks: read %.1f s, "
               "compute %.1f s, write %.1f s)\n", sets.size(), nplanes, t_total, tm[0], tm[1], tm[2]);
        printf("[PNG_Combine] read %.3f TB, wrote %.3f TB, %.1f GB/s overall\n", bt[0] / 1e12, bt[1] / 1e12,
               (bt[0] + bt[1]) / 1e9 / t_total);
        fflush(stdout);
    }
    zeldovich_ps_destroy(ps_prim);
    zeldovich_params_destroy(params);
    MPI_Finalize();
    return 0;
}
