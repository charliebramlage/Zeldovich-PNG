/*
 * Kernel_MPI: MPI/C port of 1rank-run-cosm500/kernel_MPI.py
 *
 * Reads the Gaussian potential y-slices dumped by Zeldovich_MPI (DUMP_GAUSSIAN_D),
 *     <PNG_DIR>/Phi_gaussian_y.{y}        y = 0 .. N/2
 * computes every f_NL kernel enabled in the .par file (nonzero ZD_f_NL_<shape>), one shape
 * at a time reusing the same output buffers, and writes
 *     <OUT_DIR>/kernel_<shape>_y.{y}      shape = local, equil, orthoCMB, orthoLSS
 *                                         (default OUT_DIR = PNG_DIR)
 * for the LOAD_D_FROM_FILE run. Files are keyed by y only, so the rank count here need
 * not match the Zeldovich_MPI dump or load runs.
 *
 * Usage:  mpiexec -n R Kernel_MPI [param.par] [--png-dir DIR] [--out-dir DIR] [--python-compat]
 *
 *   --python-compat                  default M = 2N, as in kernel_MPI.py (same as KERNEL_PYTHON_COMPAT=1)
 *
 * Par keys:   BoxSize, NumZRanks, ZD_f_NL_local, ZD_f_NL_equil, ZD_f_NL_orthoCMB,
 *             ZD_f_NL_orthoLSS, optional ZD_PNG_DIR (default "png_slices"),
 *             optional KernelPadM (padded grid size M)
 *             Read with ParseHeader (kernel_params.h); expand a .par2 to .par first, as Abacus
 *             does (GenParam, or scripts/par2_to_par.py -- see 1rank-run-cosm500/run.sh).
 *
 * Environment:
 *   PARAM_FILE, PNG_DIR, OUT_DIR    override the .par file / ZD_PNG_DIR / output dir (CLI args override these)
 *   KERNEL_PYTHON_COMPAT=0 (default) default M = smallest valid M >= 3N/2 (alias-free, smaller)
 *                       =1           default M = 2N, as in kernel_MPI.py (or pass --python-compat)
 *   KERNEL_PAD_M=<M>                 explicit padded size (overrides KernelPadM and the default)
 *   KERNEL_ZERO_NYQUIST=1            zero the kx/ky/kz = N/2 modes of phi before the kernels
 *   KERNEL_ONLY_TERM=<A1|C1|...>     debug: run only this term with coef 1 (output = J*Q),
 *                                    written to kernel_<term>_y.{y}
 *   KERNEL_FORCE_STAGED_A2A=1        test: always use the INT_MAX-safe staged Alltoallv path
 *
 * Precision follows precision.h (real_t / fftw_complex_t): single by default, double
 * with -Ddouble_precision=true. Slice files are float64 either way.
 *
 * Stages per kernel term: see kernel_pass.h.
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>

#include "config.h"
#include "precision.h"
#include "types.h"
#include "mpi_topology.h"

#include "kernel_grid.h"
#include "kernel_io.h"
#include "kernel_params.h"
#include "kernel_pass.h"
#include "kernel_terms.h"

static int env_int(const char *name, int dflt)
{
    const char *v = getenv(name);
    return (v != NULL && v[0] != '\0') ? atoi(v) : dflt;
}

static double peak_rss_gb(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1.0e6;   // ru_maxrss is in kB on Linux
}

#define FAIL(...) do { if (world_rank == 0) fprintf(stderr, "[Kernel_MPI] ERROR: " __VA_ARGS__); \
                       MPI_Abort(MPI_COMM_WORLD, 1); } while (0)

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int world_rank = 0, num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    double t_start = MPI_Wtime();

    // ==================================================================================
    // Arguments / environment
    // ==================================================================================
    const char *param_file = getenv("PARAM_FILE") ? getenv("PARAM_FILE") : ".runtime/param.par";
    const char *png_dir_arg = getenv("PNG_DIR");
    const char *out_dir_arg = getenv("OUT_DIR");
    int compat = env_int("KERNEL_PYTHON_COMPAT", 0);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--python-compat") == 0) {
            compat = 1;
        } else if (strcmp(argv[i], "--png-dir") == 0 && i + 1 < argc) {
            png_dir_arg = argv[++i];
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            out_dir_arg = argv[++i];
        } else if (argv[i][0] != '-') {
            param_file = argv[i];
        } else {
            FAIL("unknown argument %s\n", argv[i]);
        }
    }

    const int zero_nyquist = env_int("KERNEL_ZERO_NYQUIST", 0);
    const int pad_M_env = env_int("KERNEL_PAD_M", -1);
    const char *only_term = getenv("KERNEL_ONLY_TERM");

    // ==================================================================================
    // Rank 0: parameters, slice directories, N -> broadcast
    // ==================================================================================
    KernelParams kp;
    memset(&kp, 0, sizeof(kp));
    int hdr[2] = {0, -1};   // parse status, N
    if (world_rank == 0) {
        hdr[0] = kernel_params_load(param_file, &kp);
        if (png_dir_arg != NULL && png_dir_arg[0] != '\0') {
            snprintf(kp.png_dir, sizeof(kp.png_dir), "%s", png_dir_arg);
        }
        char path[4096 + 64];
        snprintf(path, sizeof(path), "%s/Phi_gaussian_y.0", kp.png_dir);
        hdr[1] = kernel_detect_N(path);
    }
    MPI_Bcast(hdr, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kp, sizeof(kp), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (hdr[0] != 0) FAIL("cannot read parameter file %s\n", param_file);
    if (!kp.found[0] || !kp.found[1]) FAIL("%s must define BoxSize and NumZRanks\n", param_file);
    const char *png_dir = kp.png_dir;
    const char *out_dir = (out_dir_arg != NULL && out_dir_arg[0] != '\0') ? out_dir_arg : png_dir;
    const int N = hdr[1];
    if (N <= 0) FAIL("%s/Phi_gaussian_y.0 missing or not N*N complex doubles\n", png_dir);
    if (N % 2 != 0) FAIL("N=%d must be even\n", N);
    const int num_z_ranks = (int)kp.num_z_ranks;
    if (num_z_ranks <= 0 || num_ranks % num_z_ranks != 0) {
        FAIL("num_ranks=%d is not divisible by NumZRanks=%d\n", num_ranks, num_z_ranks);
    }
    if (num_ranks > N / 2 + 1) FAIL("num_ranks=%d > N/2+1=%d\n", num_ranks, N / 2 + 1);

    int M = kernel_choose_pad_M(N, compat);
    if (kp.found[6] && kp.pad_M > 0) M = (int)kp.pad_M;
    if (pad_M_env > 0) M = pad_M_env;
    if (M < N || (M % 4) != 0 || ((M - N) % 2) != 0) {
        FAIL("padded size M=%d invalid: need M >= N=%d, M %% 4 == 0, (M - N) even\n", M, N);
    }
    if (2 * M < 3 * N && world_rank == 0) {
        fprintf(stderr, "[Kernel_MPI] WARNING: M=%d < 3N/2=%d: the quadratic product will alias\n", M, 3 * N / 2);
    }

    // f_NL selection: every nonzero of (local, equil, orthoCMB, orthoLSS) is computed, one
    // after another, each into its own kernel_<shape>_y.{y} files
    FnlType active[4];
    int num_active = 0;
    for (int i = 0; i < 4; i++) {
        if (kp.fnl[i] != 0.0) active[num_active++] = (FnlType)i;
    }
    const int debug_term = (only_term != NULL && only_term[0] != '\0');
    int only_id = -1;
    if (debug_term) {
        only_id = kernel_shape_from_name(only_term);
        if (only_id < 0) FAIL("KERNEL_ONLY_TERM=%s is not a kernel shape name\n", only_term);
        num_active = 1;   // one pass, terms set in the loop below
    }
    if (num_active == 0) FAIL("all of ZD_f_NL_local/equil/orthoCMB/orthoLSS are zero (or missing)\n");

    // ==================================================================================
    // Topology (same Cartesian grid as Zeldovich_MPI) + layout
    // ==================================================================================
    if (zd_topology_init_standalone(num_ranks, num_z_ranks) != 0) FAIL("topology init failed\n");
    const int rank = zd_cart_rank;
    const int size_x = zd_cart_size_x, size_z = zd_cart_size_z;

    KernelLayout K;
    kernel_layout_init(&K, N, M, num_ranks, rank, size_x, size_z);

    if (rank == 0) {
        printf("========================================================================\n");
        printf("Kernel_MPI: %d f_NL shape%s%s\n", num_active, num_active == 1 ? "" : "s",
               debug_term ? " [KERNEL_ONLY_TERM]" : "");
        printf("  N=%d  M=%d (padding %.3fx)  BoxSize=%g\n", N, M, (double)M / N, kp.box_size);
        printf("  ranks=%d (size_x=%d x size_z=%d), OMP threads/rank=%d\n",
               num_ranks, size_x, size_z, omp_get_max_threads());
        printf("  precision=%s, batches=%d, compat=%d, zero_nyquist=%d\n",
               PRECISION_NAME, K.num_batches, compat, zero_nyquist);
        printf("  in:  %s\n  out: %s\n", png_dir, out_dir);
        if (debug_term) {
            printf("  term: %s  coef=1\n", kernel_shape((KernelShapeId)only_id)->name);
        } else {
            for (int s = 0; s < num_active; s++) {
                KernelTerm terms[KERNEL_MAX_TERMS];
                int num_terms = kernel_terms_for(active[s], terms);
                printf("  %s:\n", fnl_type_name(active[s]));
                for (int t = 0; t < num_terms; t++) {
                    printf("    term %d: %s  coef=% .17g\n", t, kernel_shape(terms[t].shape)->name, terms[t].coef);
                }
            }
        }
        printf("========================================================================\n");
        fflush(stdout);
    }

    // ==================================================================================
    // Read this rank's primary y-slices (mirrors are built on the fly by Hermitian symmetry)
    // ==================================================================================
    const int64_t slice_elems = (int64_t)N * N;
    fftw_complex_t **phi = (fftw_complex_t**)calloc(K.my_num_pairs > 0 ? K.my_num_pairs : 1, sizeof(*phi));
    fftw_complex_t **out = (fftw_complex_t**)calloc(K.my_num_pairs > 0 ? K.my_num_pairs : 1, sizeof(*out));
    for (int b = 0; b < K.my_num_pairs; b++) {
        int y_p = kernel_batch_y_p(&K, rank, b);
        char path[4096 + 64];
        snprintf(path, sizeof(path), "%s/Phi_gaussian_y.%d", png_dir, y_p);
        phi[b] = kernel_alloc_complex(slice_elems, "phi slice");
        int err = kernel_read_slice(path, N, phi[b]);
        if (err != 0) {
            fprintf(stderr, "[Kernel_MPI] ERROR: rank %d cannot read %s (%s)\n",
                    rank, path, err == 1 ? "missing" : "wrong size");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (y_p == 0 || y_p == N / 2) kernel_fix_self_conjugate_plane(N, phi[b]);
        if (zero_nyquist) kernel_zero_nyquist(N, y_p, phi[b]);

        out[b] = kernel_alloc_complex(slice_elems, "output slice");   // zeroed per shape below
    }

    // ==================================================================================
    // Kernel terms
    // ==================================================================================
    KernelPass P;
    kernel_pass_init(&P, &K, zd_comm_2d, kp.box_size, phi, out);

    if (rank == 0) {
        double gb = 1.0e-9 * sizeof(fftw_complex_t);
        printf("[Memory, rank 0] recv_buffer (also holds q_half) %.3f GB, phi+out %.3f GB\n",
               K.recv_total * gb, 2.0 * K.my_num_pairs * slice_elems * gb);
        fflush(stdout);
    }

    if (rank == 0) {
        if (mkdir(out_dir, 0755) != 0 && errno != EEXIST) {
            fprintf(stderr, "[Kernel_MPI] ERROR: cannot create %s\n", out_dir);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(zd_comm_2d);

    // One f_NL shape at a time: phi is read-only in run_kernel_term, so the same phi and
    // out slices are reused for every shape and peak memory matches a single-shape run
    double t_terms = MPI_Wtime();
    for (int s = 0; s < num_active; s++) {
        KernelTerm terms[KERNEL_MAX_TERMS];
        int num_terms;
        const char *label;
        if (debug_term) {
            terms[0] = (KernelTerm){(KernelShapeId)only_id, 1.0};
            num_terms = 1;
            label = kernel_shape((KernelShapeId)only_id)->name;
        } else {
            num_terms = kernel_terms_for(active[s], terms);
            label = fnl_type_name(active[s]);
        }

        for (int b = 0; b < K.my_num_pairs; b++) {
            memset(out[b], 0, sizeof(fftw_complex_t) * (size_t)slice_elems);
        }

        double t_shape = MPI_Wtime();
        for (int t = 0; t < num_terms; t++) {
            const KernelShape *shape = kernel_shape(terms[t].shape);
            double t0 = MPI_Wtime();
            run_kernel_term(&P, shape, terms[t].coef);
            MPI_Barrier(zd_comm_2d);
            if (rank == 0) {
                printf("[Kernel_MPI] %s: term %d/%d (%s) done in %.3f s\n",
                       label, t + 1, num_terms, shape->name, MPI_Wtime() - t0);
                fflush(stdout);
            }
        }

        // Rank r writes exactly the y_primary slices it read; files keyed by y
        for (int b = 0; b < K.my_num_pairs; b++) {
            int y_p = kernel_batch_y_p(&K, rank, b);
            char path[4096 + 64];
            snprintf(path, sizeof(path), "%s/kernel_%s_y.%d", out_dir, label, y_p);
            if (kernel_write_slice(path, N, out[b]) != 0) {
                fprintf(stderr, "[Kernel_MPI] ERROR: rank %d cannot write %s\n", rank, path);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        MPI_Barrier(zd_comm_2d);
        if (rank == 0) {
            printf("[Kernel_MPI] shape %d/%d (%s) done in %.3f s -> %s/kernel_%s_y.*\n",
                   s + 1, num_active, label, MPI_Wtime() - t_shape, out_dir, label);
            fflush(stdout);
        }
    }
    t_terms = MPI_Wtime() - t_terms;

    // ==================================================================================
    // Timing / memory summary (max over ranks)
    // ==================================================================================
    double local_t[7] = {P.t_fill_fft2d, P.t_pack, P.t_comm_fwd, P.t_zslab, P.t_comm_ret,
                         P.t_fft2d_accum, peak_rss_gb()};
    double max_t[7];
    MPI_Reduce(local_t, max_t, 7, MPI_DOUBLE, MPI_MAX, 0, zd_comm_2d);
    if (rank == 0) {
        printf("[Timing, max over ranks] fill+2D bwd %.3f s | pack %.3f s | Alltoallv fwd %.3f s | "
               "z-slab 1D+multiply %.3f s | Alltoallv ret %.3f s | 2D fwd+accumulate %.3f s\n",
               max_t[0], max_t[1], max_t[2], max_t[3], max_t[4], max_t[5]);
        printf("[Kernel_MPI] kernel terms %.3f s, total %.3f s, peak RSS (max rank) %.3f GB\n",
               t_terms, MPI_Wtime() - t_start, max_t[6]);
        fflush(stdout);
    }

    kernel_pass_free(&P);
    for (int b = 0; b < K.my_num_pairs; b++) { free(phi[b]); free(out[b]); }
    free(phi);
    free(out);
    kernel_layout_free(&K);
    zd_topology_cleanup();
    MPI_Finalize();
    return 0;
}
