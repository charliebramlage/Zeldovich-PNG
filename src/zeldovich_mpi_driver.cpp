/*
 * 1. INITIALIZATION
 *    - MPI_Init; parse param_file (required)
 *    - Read NP from param file, derive ppd (= N), validate N; validate num_ranks vs total_pairs
 *    - zd_comm_2d is a derived communicator with reorder=1, so MPI may remap ranks for topology locality
*     - ex: world_rank=7 may be rank=3 in zd_comm_2d.
*     - broadcast steps use MPI_COMM_WORLD + world_rank (all processes must participate)
*     - most decomposition/compute logic uses zd_comm_2d + rank (grid-aware topology)
 *
 * 2. Y-SLICE PAIR DISTRIBUTION
 *    - total_pairs = N/2 + 1 (conjugate pairs: (0,0), (1,N-1), ..., N/2 self-conj)
 *    - Distribute pairs across ranks; build my_pair_list for this rank
 *
 * 3. PARAMETER AND SPECTRUM LOADING
 *    - Load zeldovich-PLT params; create power spectrum (ZD_Pk_filename -> file spline, else ZD_Pk_powerlaw_index -> power law)
 *    - Load PLT eigenmodes if qPLT; set narray (1/2/4 by qdensity/qPLT)
 *
 * 4. FFT SETUP
 *    - setup_fftw_plans_full(plan_2d, plan_1d_y) on a temporary N×N plan buffer;
 *      free plan buffer before recv_buffer alloc (Option A: no stage/recv overlap at peak)
 *
 * 5. GRID DECOMPOSITION
 *    - get_extended_grid_bounds; my_pencils for this rank’s XZ region
 *
 * 6. PRE-COMPUTATION (V11 PERSISTENT RECV_BUFFER)
 *    - global_max_batches; src_total_slices per source; prefix-sum recv_displs_src
 *    - Allocate persistent recv_buffer (source-grouped: [src][y][pencil][array])
 *    - Build y_owner_src, y_src_local_idx, y_batch_idx, y_slice_idx_in_batch
 *
 * 7. MULTI-BATCH LOOP (for each batch)
 *    a. Get batch’s (y_primary, y_mirror)
 *    b. Alloc local_y_slices + fft_stage_2d → generate + 2D FFT → free stage
 *    c. calculate_batch_send_recv_counts; pack_slices_to_send_buffer; free local_y_slices
 *    d. zd_alltoallv_int64(send_buffer -> recv_buffer): MPI_Alltoallv_c on MPI >= 4, Isend/Irecv otherwise
 *    e. Update src_write_cursor; free per-batch send buffer
 *
 * 8. Z-SLAB STREAMING (for each Z owned by this rank)
 *    - Allocate local_z_slab (one Z-slab: [Array][X][Y])
 *    - For each z: z_streaming_unpack(recv_buffer -> local_z_slab, staged 1D FFT in Y)
 *    - Write output: PARTICLE_OUTPUT_MODE 0 -> WriteParticlesSlab_range
 *                    PARTICLE_OUTPUT_MODE 1 -> .bin files
 *                    PARTICLE_OUTPUT_MODE 2 -> .bin then read-back -> WriteParticlesSlab_range
 *                    PARTICLE_OUTPUT_MODE 3 -> CPD-slab-ordered streaming append (one file per slab, optionally split by z-rank)
 *                    PARTICLE_OUTPUT_MODE 4 -> z-slab streaming append (one file per z-slab, in x-rank subdirs)
 *
 * 9. CLEANUP (plans, buffers)
 *    - Free plans, recv_buffer, local buffers, params, ps, PLT eigenmodes
 *    - MPI_Finalize
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <limits.h>
#include <string>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <omp.h> 
#include <fftw3.h>
#include <sys/stat.h>
#include <errno.h>
#include <execinfo.h>  
#include <unistd.h>    // For getpid
#ifdef __linux__
#include <sched.h>     // sched_getcpu() — OpenMP thread CPU placement diagnostics
#endif

// Include PCG RNG and STimer (vendored from zeldovich-PLT)
#include "pcg-rng/pcg_random.hpp"
#include <STimer.h>
#include <ParseHeader.hh>

// --- CONFIGURATION AND TYPES (config.h, precision.h, types.h) ---
#include "config.h"
#include "precision.h"
#include "types.h"

// CXI/libfabric requires page-aligned (base, len) for memory registration.
// Round allocation size up to page multiple to avoid cxil_map EINVAL.
#define PAGE_SIZE_CXI 4096
#define ROUND_UP_PAGE(x) ((((size_t)(x)) + PAGE_SIZE_CXI - 1) & ~((size_t)(PAGE_SIZE_CXI - 1)))

// --- UTILITIES ---
#include "utils/printing.h"
#include "utils/verification.h"
#include "utils/decomposition.h"
#include "utils/batch_helpers.h"

// --- CORE MODULES ---
#include "fft/fft_setup.h"
#include "generation/ZD_MPI_generation.h"
#include "communication/mpi_exchange.h"
#include "streaming/z_streaming.h"

// --- PCG RNG MODULE (zeldovich-PLT power spectrum / cgauss) ---
#include "utils/rng.h"
#include "utils/zeldovich_wrapper.h"  
#include "utils/plt_eigenmodes.h"    
#include "output/output_new.h"

// --- MPI TOPOLOGY ---
#include "mpi_topology.h"
#include "ic_embed_flags.h"

/** Rank 0: print MPI thread level from MPI_Init_thread (required vs provided). */
static void print_mpi_init_thread_levels(int required, int provided, int world_rank)
{
    if (world_rank != 0) return;
    printf("[MPI] MPI_Init_thread: required=%d provided=%d\n", required, provided);
    fflush(stdout);
}

static int broadcast_parameter_header_bytes(
    const char *param_file,
    int world_rank,
    MPI_Comm comm,
    std::vector<char> &header_bytes
) {
    uint64_t header_len = 0;
    if (world_rank == 0) {
        HeaderStream hs{fs::path(param_file)};
        hs.ReadHeader();
        if (hs.buffer == NULL || hs.bufferlength < 2) {
            fprintf(stderr, "ERROR: Invalid parameter header read from %s\n", param_file);
            return 1;
        }
        if (hs.bufferlength > static_cast<size_t>(INT_MAX)) {
            fprintf(stderr, "ERROR: Parameter header too large for MPI_Bcast count: %zu\n", hs.bufferlength);
            return 1;
        }
        header_bytes.assign(hs.buffer, hs.buffer + hs.bufferlength);
        header_len = static_cast<uint64_t>(hs.bufferlength);
    }

    MPI_Bcast(&header_len, 1, MPI_UINT64_T, 0, comm);
    if (header_len < 2) {
        if (world_rank == 0) {
            fprintf(stderr, "ERROR: Broadcast parameter header length is invalid: %llu\n",
                    static_cast<unsigned long long>(header_len));
        }
        return 1;
    }
    if (header_len > static_cast<uint64_t>(INT_MAX)) {
        if (world_rank == 0) {
            fprintf(stderr, "ERROR: Broadcast parameter header exceeds MPI_Bcast int count: %llu\n",
                    static_cast<unsigned long long>(header_len));
        }
        return 1;
    }

    if (world_rank != 0) {
        header_bytes.resize(static_cast<size_t>(header_len));
    }
    MPI_Bcast(header_bytes.data(), static_cast<int>(header_len), MPI_BYTE, 0, comm);
    return 0;
}

/**
 * If ZD_MPI_PRINT_AFFINITY is set (non-empty, not "0"), print one line per OpenMP thread
 * with sched_getcpu(). Linux only. Can produce many lines (ranks x threads); use with small jobs
 * or redirect stdout. Compare with mpiexec --display-bindings (Open MPI).
 */
static void print_omp_thread_cpus_if_requested(int world_rank)
{
    const char *env = getenv("ZD_MPI_PRINT_AFFINITY");
    if (env == NULL || env[0] == '\0') return;
    if (env[0] == '0' && env[1] == '\0') return;
#ifdef __linux__
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int cpu = sched_getcpu();
        printf("[AFFINITY] rank %d thread %d cpu %d\n", world_rank, tid, cpu);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
#else
    if (world_rank == 0) {
        fprintf(stderr,
                "[AFFINITY] ZD_MPI_PRINT_AFFINITY is set but sched_getcpu() is only used on Linux; "
                "skipped.\n");
        fflush(stderr);
    }
#endif
}

// --- DRIVER: MPI must already be initialized (standalone main or Abacus host) ---
extern "C" int zeldovich_mpi_driver_run(int argc, char **argv)
{
    int required = MPI_THREAD_SINGLE;
    int provided = MPI_THREAD_SINGLE;
    MPI_Query_thread(&provided);

    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    print_mpi_init_thread_levels(required, provided, world_rank);
    print_omp_thread_cpus_if_requested(world_rank);

    // ========================================================================
    // Stage 1: Parse arguments (before topology so we can load params for CPD/PPD)
    // ========================================================================
    int N = 64;
    const char* param_file = NULL;

    if (argc < 2) {
        if (world_rank == 0) {
            fprintf(stderr, "Usage: %s param_file.par\n", argv[0]);
            fprintf(stderr, "  param_file.par: zeldovich-PLT parameter file (required)\n");
        }
        if (!zeldovich_ic_embedded) {
            MPI_Finalize();
        }
        return 1;
    }

    if (argc >= 2) {
        param_file = argv[1];
    }
    if (param_file == NULL) {
        if (world_rank == 0) {
            fprintf(stderr, "Error: Parameter file required \n");
            fprintf(stderr, "Usage: %s param_file.par\n", argv[0]);
        }
        if (!zeldovich_ic_embedded) {
            MPI_Finalize();
        }
        return 1;
    }

    // ========================================================================
    // Load param file early for CPD/PPD (used for CPD-aligned MPI grid)
    // ========================================================================
    ParametersHandle params = NULL;
    std::vector<char> param_header_bytes;
    if (broadcast_parameter_header_bytes(param_file, world_rank, MPI_COMM_WORLD, param_header_bytes) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    params = zeldovich_params_create_from_buffer(
        param_header_bytes.data(),
        param_header_bytes.size(),
        param_file
    );
    if (!params) {
        if (world_rank == 0) {
            fprintf(stderr, "Failed to load param file: %s\n", param_file);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int cpd = zeldovich_params_get_cpd(params);
    int64_t ppd_from_file = zeldovich_params_get_ppd(params);
    N = (int)ppd_from_file;
    if (world_rank == 0) {
        fprintf(stdout, "Using ppd=%ld (N=%d) derived from NP in param file.\n",
                (long)ppd_from_file, N);
    }

    // ========================================================================
    // MPI Cartesian Topology Setup (grid_z from param file, grid_x from MPI size)
    // ========================================================================
    int grid_x, grid_z;
    // old code:
    // grid_z = zeldovich_params_get_NumZRanks(params);
    // grid_x = num_ranks / grid_z; // set equal to number abacus zranks 
    
    // make this change:
    grid_x = abacus_params_get_NumZRanks(params); // third transpose! before it was ZD z ranks!
    grid_z = num_ranks / grid_x;

    // When integrated in abacus, InitParallelTopology() in multistep will
    // assert that grid_z == MPI_SIZE/NumZRanks.

    // old code:
    // if (num_ranks % grid_z != 0) {
    //     if (world_rank == 0) {
    //         fprintf(stderr, "Error: num_ranks=%d is not evenly divisible by ZD_NumZRanks=%d.\n",
    //                 num_ranks, grid_z);
    //     }
    //     MPI_Abort(MPI_COMM_WORLD, 1);
    // }

    // new change:
    if (num_ranks % grid_x != 0) {
        if (world_rank == 0) {
            fprintf(stderr, "Error: num_ranks=%d is not evenly divisible by NumZRanks=%d.\n",
                    num_ranks, grid_x);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // old code:
    // int dims[2] = { grid_x, grid_z };

    // new change:
    int dims[2] = { grid_z, grid_x }; //made this change
    int periodic[2] = { 1, 1 };
    int reorder = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, reorder, &zd_comm_2d);

    if (zd_comm_2d == MPI_COMM_NULL) {
        if (world_rank == 0) {
            fprintf(stderr, "Error: unable to create MPI Cartesian grid with grid_x=%d, grid_z=%d.\n",
                    grid_x, grid_z);
            fprintf(stderr, "       Check that the supplied grid dimensions are valid for this MPI launch.\n");
        }
        if (!zeldovich_ic_embedded) {
            MPI_Finalize();
        }
        return 1;
    }

    MPI_Comm_set_errhandler(zd_comm_2d, MPI_ERRORS_RETURN);

    int rank;
    MPI_Comm_rank(zd_comm_2d, &rank);

    int coords[2];
    MPI_Cart_coords(zd_comm_2d, rank, 2, coords);
    int rank_x = coords[0];
    int rank_z = coords[1];

    if (rank == 0) {
        printf("========================================================================\n");
        printf("MPI Cartesian Topology Initialized (parameter-driven)\n");
        printf("  Grid: %d x %d = %d ranks (cpd=%d, N=%d)\n", grid_x, grid_z, num_ranks, cpd, N);
        printf("  Periodic: [X=%s, Z=%s]\n",
               periodic[0] ? "yes" : "no", periodic[1] ? "yes" : "no");
        printf("  Reorder: %s (hardware-aware rank assignment)\n",
               reorder ? "enabled" : "disabled");
        printf("========================================================================\n");
    }

    printf("[Rank %d] Cartesian coords: (rank_x=%d, rank_z=%d)\n", rank, rank_x, rank_z);

    (void)rank_x;
    (void)rank_z;

    // ========================================================================
    // Stage 2: Initialization
    // ========================================================================
    
    int total_pairs = N/2 + 1;  // Number of Y-slice pairs to distribute
    
    // Check rank count is valid
    if (num_ranks < 1 || num_ranks > total_pairs) {
        if (rank == 0) {
            fprintf(stderr, "Error: Invalid num_ranks=%d for N=%d (total_pairs=%d)\n",
                    num_ranks, N, total_pairs);
        }
        if (!zeldovich_ic_embedded) {
            MPI_Finalize();
        }
        return 1;
    }

    if (rank == 0) {
        printf("N=%d ranks=%d\n", N, num_ranks);
    }
    
    // Struct to hold a pair of Y-slices (primary + conjugate)
    typedef struct {
        int y_primary;    
        int y_mirror;     // may equal y_primary if self-conjugate
        bool is_self_conjugate;  
    } YSlicePair;
    
    // Distribute pairs across ranks
    int pairs_per_rank = total_pairs / num_ranks;
    int remainder = total_pairs % num_ranks;
    
    int my_num_pairs;
    int my_pair_start;
    
    if (rank < remainder) {
        my_num_pairs = pairs_per_rank + 1;
        my_pair_start = rank * (pairs_per_rank + 1);
    } else {
        my_num_pairs = pairs_per_rank;
        my_pair_start = remainder * (pairs_per_rank + 1) + (rank - remainder) * pairs_per_rank;
    }
    
    // Allocate memory for the list of pairs assigned to this rank
    YSlicePair *my_pair_list = NULL; // Array of YSlicePair structs
    bool is_idle_rank = (my_num_pairs == 0); // no pairs assigned to rank
    
    if (!is_idle_rank) {
        my_pair_list = (YSlicePair*)malloc(sizeof(YSlicePair) * my_num_pairs);
        
        for (int i = 0; i < my_num_pairs; i++) {
            int pair_idx = my_pair_start + i;
            int y_primary = pair_idx;
            int y_mirror = (y_primary == 0 || y_primary == N/2) ? y_primary : N - y_primary;
            
            my_pair_list[i].y_primary = y_primary;
            my_pair_list[i].y_mirror = y_mirror;
            my_pair_list[i].is_self_conjugate = (y_primary == y_mirror);
        }
    } else {
        if (rank == num_ranks - 1 || (rank < num_ranks && rank + 1 >= num_ranks)) {
            printf("[WARNING] Rank %d is y-slice IDLE (total_pairs=%d, num_ranks=%d)\n",
                   rank, total_pairs, num_ranks);
        }
    }
    
    int num_my_slices = 0;
    fftw_complex_t *local_y_slices = NULL;
    fftw_complex_t *fft_stage_2d = NULL;
    fftw_complex_t *local_z_slab = NULL;
    ExtendedGridBounds my_extended_bounds;
    int my_pencils = 0;

    if (is_idle_rank) {
        num_my_slices = 0;
        my_extended_bounds.core.x_start = my_extended_bounds.core.x_end = 0;
        my_extended_bounds.core.z_start = my_extended_bounds.core.z_end = 0;
        my_extended_bounds.padded.x_start = my_extended_bounds.padded.x_end = 0;
        my_extended_bounds.padded.z_start = my_extended_bounds.padded.z_end = 0;
        my_extended_bounds.num_pencils_core = 0;
        my_extended_bounds.num_pencils_padded = 0;
        my_pencils = 0;
    }
    
    PowerSpectrumHandle ps = NULL;
    PowerSpectrumHandle ps_primordial = NULL;

    // Params already loaded earlier for CPD-aligned grid
    if (params != NULL) {
        if (rank == 0) {
            printf("[INIT] Using zeldovich-PLT parameters from: %s (cpd=%d)\n", param_file, cpd);
            printf("[INIT] InitialConditionsDirectory: %s\n",
                   static_cast<ZeldovichParameters*>(params)->output_dir.string().c_str());
            fflush(stdout);
        }

        uint64_t seed = (uint64_t)zeldovich_params_get_seed(params);
        initialize_global_pcg(N, N, N, seed);

        SetupOutputDir(*static_cast<ZeldovichParameters*>(params));
        InitOutputBuffers(*static_cast<ZeldovichParameters*>(params));

        // Create PowerSpectrum object (each rank creates its own)
        // Spline resolution: configured via SPLINE_RESOLUTION (default 128)
        ps = zeldovich_ps_create(SPLINE_RESOLUTION, params);
        if (!ps) {
            if (rank == 0) {
                fprintf(stderr, "ERROR: Failed to create PowerSpectrum object\n");
            }
            MPI_Abort(zd_comm_2d, 1);
        }
        
        const char* pk_file = zeldovich_params_get_Pk_filename(params);
        if (pk_file != NULL && pk_file[0] != '\0') {
            // every rank ends up w/ 2 local recv buffers for pk: 
            // pk_k (length n_pk); pk_P (length n_pk)
            // each rank calls zeldovich_ps_init_from_raw_pk()
            // to run spline() and normalization --> after which ps is ready for cguass, etc..

            uint64_t n_pk = 0;
            std::vector<double> pk_k;
            std::vector<double> pk_P;
            if (world_rank == 0) {
                if (zeldovich_pk_load_text_file_vectors(pk_file, params, pk_k, pk_P) != 0) {
                    fprintf(stderr, "ERROR: Failed to read power spectrum file: %s\n", pk_file);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                n_pk = (uint64_t)pk_k.size();
                if (n_pk == 0) {
                    fprintf(stderr, "ERROR: Power spectrum file has no valid rows: %s\n", pk_file);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            MPI_Bcast(&n_pk, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
            if (world_rank != 0) {
                pk_k.resize((size_t)n_pk);
                pk_P.resize((size_t)n_pk);
            }

            int n_pk_i = (int)n_pk;
            if (n_pk_i > 0) {
                MPI_Bcast(pk_k.data(), n_pk_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pk_P.data(), n_pk_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
            if (zeldovich_ps_init_from_raw_pk(ps, pk_k.data(), pk_P.data(), (size_t)n_pk, params) != 0) {
                if (rank == 0) {
                    fprintf(stderr, "ERROR: Failed to initialize power spectrum from broadcast P(k) table\n");
                }
                MPI_Abort(zd_comm_2d, 1);
            }
            if (rank == 0) {
                printf("[INIT] Power spectrum initialized from file (rank-0 read + MPI broadcast): %s\n",
                       pk_file);
            }
        } else {
            double powerlaw_index = zeldovich_params_get_Pk_powerlaw_index(params);
            if (powerlaw_index == 1000.0) { // ZD_Pk_powerlaw_index == 1000 means “not set”
                if (rank == 0) {
                    fprintf(stderr, "ERROR: Set ZD_Pk_filename or ZD_Pk_powerlaw_index (mutually exclusive)\n");
                }
                MPI_Abort(zd_comm_2d, 1);
            }
            if (zeldovich_ps_init_powerlaw(ps, powerlaw_index, params) != 0) {
                if (rank == 0) {
                    fprintf(stderr, "ERROR: Failed to initialize power spectrum (power law index: %.2f)\n", powerlaw_index);
                }
                MPI_Abort(zd_comm_2d, 1);
            }
        }

        // ====================================================================================
        // Primordial power spectrum (ZD_Pk_primordial_filename, default
        // "class_pk_primordial_dimensional.dat"), loaded the same way as the main P(k) file
        // above: rank 0 reads, all ranks receive the data via MPI_Bcast, each rank builds
        // its own spline from the broadcast arrays.
        //
        // This is OPTIONAL: unlike the main spectrum, a missing/unreadable file here just
        // disables the primordial lookup (ps_primordial stays NULL) rather than aborting --
        // generate_zd_mpi_slice_pair_local already treats a NULL handle as "P_primordial=0"
        // for callers that don't need this feature.
        // ====================================================================================
        {
            // Used exactly as given in the .par file (relative or absolute) -- same
            // convention as ZD_Pk_filename itself, which has no directory-joining logic
            // either (ReadPkTextFileIntoVectors just fopen()s whatever string it's given).
            const char* primordial_filename_param = zeldovich_params_get_Pk_primordial_filename(params);
            std::string primordial_path = (primordial_filename_param != NULL && primordial_filename_param[0] != '\0')
                ? std::string(primordial_filename_param)
                : "class_pk_primordial_dimensional.dat";  // fallback if somehow empty

            uint64_t n_prim = 0;
            std::vector<double> prim_k, prim_P;
            int prim_read_ok = 1;
            if (world_rank == 0) {
                if (zeldovich_pk_load_text_file_vectors(primordial_path.c_str(), params, prim_k, prim_P) != 0
                    || prim_k.empty()) {
                    fprintf(stderr,
                        "[WARNING] Failed to read primordial power spectrum file: %s (primordial lookup disabled)\n",
                        primordial_path.c_str());
                    prim_read_ok = 0;
                } else {
                    n_prim = (uint64_t)prim_k.size();
                }
            }
            MPI_Bcast(&prim_read_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (prim_read_ok) {
                MPI_Bcast(&n_prim, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
                if (world_rank != 0) {
                    prim_k.resize((size_t)n_prim);
                    prim_P.resize((size_t)n_prim);
                }
                int n_prim_i = (int)n_prim;
                if (n_prim_i > 0) {
                    MPI_Bcast(prim_k.data(), n_prim_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(prim_P.data(), n_prim_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                    ps_primordial = zeldovich_ps_create(n_prim_i, params);
                    if (ps_primordial != NULL) {
                        // normalization = 1.0 / boxsize^3 and nothing else: only the
                        // FFT/box-volume convention factor shared with the main spectrum,
                        // never its sigma(R)/growth rescaling or Pk_smooth
                        int prim_init_rc = zeldovich_ps_init_primordial_from_raw_pk(
                            ps_primordial, prim_k.data(), prim_P.data(), (size_t)n_prim, params);

                        if (prim_init_rc != 0) {
                            if (rank == 0) {
                                fprintf(stderr,
                                    "[WARNING] Failed to initialize primordial power spectrum "
                                    "(primordial lookup disabled)\n");
                            }
                            zeldovich_ps_destroy(ps_primordial);
                            ps_primordial = NULL;
                        } else {
                            if (rank == 0) {
                                printf("[INIT] Primordial power spectrum initialized from file "
                                       "(rank-0 read + MPI broadcast): %s\n", primordial_path.c_str());
                            }
                        }
                    }
                }
            }
        }
        
        // Load PLT eigenmodes from file
        int qPLT = zeldovich_params_get_qPLT(params);
        if (qPLT) {

            const char* PLT_filename = zeldovich_params_get_PLT_filename(params);
            const char* ICFormat = zeldovich_params_get_ICFormat(params);

            // Validate ICFormat starts with "RV" when qPLT is enabled
            if (!ICFormat || strncmp(ICFormat, "RV", 2) != 0) {
                if (rank == 0) {
                    fprintf(stderr, "ERROR: qPLT is enabled but ICFormat does not start with 'RV'\n");
                    fprintf(stderr, "       Current ICFormat: '%s'\n", ICFormat ? ICFormat : "(empty)");
                    fprintf(stderr, "       Set 'ICFormat = RV' or 'ICFormat = RVDoubleZel' in parameter file\n");
                }
                MPI_Abort(zd_comm_2d, 1);
            }
            
            uint64_t plt_nbytes = 0;
            std::vector<uint8_t> plt_buf;
            if (world_rank == 0) {
                FILE *eigf = fopen(PLT_filename, "rb");
                if (!eigf) {
                    fprintf(stderr, "ERROR: Could not open PLT eigenmode file: %s\n", PLT_filename);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (fseek(eigf, 0, SEEK_END) != 0) {
                    fprintf(stderr, "ERROR: Could not seek PLT eigenmode file: %s\n", PLT_filename);
                    fclose(eigf);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                long psz = ftell(eigf);
                if (psz < 0) {
                    fprintf(stderr, "ERROR: Could not stat PLT eigenmode file: %s\n", PLT_filename);
                    fclose(eigf);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                rewind(eigf);
                plt_buf.resize((size_t)psz);
                if (psz > 0 && fread(plt_buf.data(), 1, (size_t)psz, eigf) != (size_t)psz) {
                    fprintf(stderr, "ERROR: Could not read PLT eigenmode file: %s\n", PLT_filename);
                    fclose(eigf);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                fclose(eigf);
                plt_nbytes = (uint64_t)psz;
            }

            // broadcast length first, then bytes
            MPI_Bcast(&plt_nbytes, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
            if (world_rank != 0) {
                plt_buf.resize((size_t)plt_nbytes);
            }
            if (plt_nbytes > (uint64_t)INT_MAX) {
                if (rank == 0) {
                    fprintf(stderr, "ERROR: PLT eigenmode file too large for MPI_Bcast byte count\n");
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (plt_nbytes > 0u) {
                MPI_Bcast(plt_buf.data(), (int)plt_nbytes, MPI_BYTE, 0, MPI_COMM_WORLD);
            }
            if (plt_load_eigenmodes_from_buffer(plt_buf.data(), (size_t)plt_nbytes) != 0) {
                if (rank == 0) {
                    fprintf(stderr, "ERROR: Failed to parse PLT eigenmodes after broadcast (source file: %s)\n",
                            PLT_filename);
                }
                MPI_Abort(zd_comm_2d, 1);
            }

            if (rank == 0) {
                printf("[INIT] Loaded PLT eigenmodes from file (rank-0 read + MPI bcast): %s\n",
                       PLT_filename);
            }
        }
    }

    // Determine the number of arrays
    int narray;
    if (params != NULL) {
        int qdensity = zeldovich_params_get_qdensity(params);
        
        if (qdensity == 2) {
            narray = 1;
        } else {
            int qPLT = zeldovich_params_get_qPLT(params);
            narray = qPLT ? 4 : 2;
        }
    } else {
        // No parameter file: use compile-time default
        narray = NARRAY;
    }

    if (rank == 0) {
        printf("[Stage 2] Complete.\n");
    }
    
    // ==================================================================================
    // Temporary plan buffer for 2D FFT setup only
    // local_y_slices: per-batch alloc before generate, free after pack (before MPI).
    // fft_stage_2d: per-batch alloc around generate+2D FFT, freed before pack.
    // ==================================================================================
    size_t fft_stage_bytes = (size_t)N * (size_t)N * sizeof(fftw_complex_t);
    size_t local_y_slices_bytes = 0;
    fftw_complex_t *fft_plan_buffer = NULL;

    if (!is_idle_rank) {
        local_y_slices_bytes =
            (size_t)2 * (size_t)narray * (size_t)N * (size_t)N * sizeof(fftw_complex_t);
        if (posix_memalign((void**)&fft_plan_buffer, ALIGN_BYTES, fft_stage_bytes) != 0) {
            fprintf(stderr, "Rank %d: posix_memalign failed for fft_plan_buffer\n", rank);
            MPI_Abort(zd_comm_2d, 1);
        }
        // parallel first touch (FFTW_MEASURE may overwrite during planning)
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < N; y++) {
            fftw_complex_t *row = fft_plan_buffer + (size_t)y * (size_t)N;
            memset(row, 0, (size_t)N * sizeof(fftw_complex_t));
        }
    }

    const char *local_wisdom_dir = "/dev/shm/Abacus_wisdom";
    if (params != NULL) {
        const char *parsed_local_wisdom_dir = zeldovich_params_get_local_wisdom_dir(params);
        if (parsed_local_wisdom_dir != NULL && parsed_local_wisdom_dir[0] != '\0') {
            local_wisdom_dir = parsed_local_wisdom_dir;
        }
    }

    // ========================================================================
    // STAGE 3: SETUP FFT PLANS 
    // ========================================================================
    // plan_2d: FFTW_PLAN_DFT_2D on fft_plan_buffer (temporary; freed before recv_buffer)
    fftw_complex_t *plan_buffer = (!is_idle_rank && fft_plan_buffer != NULL) ? fft_plan_buffer : nullptr;
    fftw_plan_t plan_2d, plan_1d_y; // setup both plans
    setup_fftw_plans_full(N, narray, plan_buffer, &plan_2d, &plan_1d_y, local_wisdom_dir);

    if (fft_plan_buffer != NULL) {
        free(fft_plan_buffer);
        fft_plan_buffer = NULL;
    }
    
    // Grid bounds: CPD-aligned when params/cpd present
    if (!is_idle_rank) {
        if (params != NULL) {
            my_extended_bounds = get_extended_grid_bounds_CPD_aligned(rank, N, num_ranks, grid_x, grid_z, cpd);
        } else {
            my_extended_bounds = get_extended_grid_bounds(rank, N, num_ranks, grid_x, grid_z);
        }
#if USE_X_PADDING
        my_pencils = my_extended_bounds.num_pencils_padded;
#else
        my_pencils = my_extended_bounds.num_pencils_core;
#endif
        
    }

    // ========================================================================
    // STAGE 4: METADATA CALCULATION AND ALLOCATION
    // ========================================================================    
    int max_batches = 0;
    if (!is_idle_rank && my_num_pairs > max_batches) {
        max_batches = my_num_pairs;
    }
    int global_max_batches = 0;
    MPI_Allreduce(&max_batches, &global_max_batches, 1, MPI_INT, MPI_MAX, zd_comm_2d);
    
    // Calculate per-source totals and allocate persistent recv_buffer
    int *src_total_slices = NULL;       // [num_ranks] - total Y-slices from each source
    int64_t *recv_displs_src = NULL;    // [num_ranks] - base offset per source (elements)
    int64_t *src_write_cursor = NULL;   // [num_ranks] - current write position per source
    int *y_owner_src = NULL;            // [N] - which rank generated Y=i
    int *y_src_local_idx = NULL;        // [N] - index within that source's chunk
    int *y_batch_idx = NULL;            // [N] - which batch Y=i came from (for unpacking)
    int *y_slice_idx_in_batch = NULL;   // [N] - slice_idx (0 or 1) in that batch (for unpacking)
    int **src_batch_slice_counts = NULL; // [num_ranks][global_max_batches] - slices per batch per source
    fftw_complex_t *recv_buffer = NULL; // Persistent recv buffer (source-grouped layout)
    int64_t recv_total_elems = 0;       // Total elements in recv_buffer

    src_total_slices = (int*)calloc(num_ranks, sizeof(int));
    
    // Count how many Y-slices each source will send across ALL batches
    for (int src = 0; src < num_ranks; src++) {
        for (int batch = 0; batch < global_max_batches; batch++) {
            int count = get_rank_batch_slice_count(src, batch, N, num_ranks);
            src_total_slices[src] += count;
        }
    }

    // Compute receive displacements (pre fix the sum)
    recv_displs_src = (int64_t*)malloc(sizeof(int64_t) * num_ranks);
    recv_total_elems = 0;
    
    for (int src = 0; src < num_ranks; src++) {
        recv_displs_src[src] = recv_total_elems;
        recv_total_elems += (int64_t)src_total_slices[src] * my_pencils * narray;
    }
    
    // Allocate persistent recv_buffer: [src_rank][y_idx_for_src][pencil_idx][array_idx]
    // Round size up to page multiple for CXI/libfabric memory registration (avoids cxil_map EINVAL).
    if (!is_idle_rank && recv_total_elems > 0) {
        size_t recv_bytes = sizeof(fftw_complex_t) * (size_t)recv_total_elems;
        size_t recv_alloc = ROUND_UP_PAGE(recv_bytes);
        if (posix_memalign((void**)&recv_buffer, ALIGN_BYTES, recv_alloc) != 0) {
            fprintf(stderr, "Rank %d: posix_memalign failed for recv_buffer\n", rank);
            MPI_Abort(zd_comm_2d, 1);
        }
        
        memset(recv_buffer, 0, recv_bytes);
        if (rank == 0) {
            fprintf(stdout, "[MPI-DIAG] recv_buffer: %zu elems, %.3f GB allocated (%.3f GB rounded)\n",
                    (size_t)recv_total_elems, recv_bytes / 1.0e9, recv_alloc / 1.0e9);
            fflush(stdout);
        }
    }
    
    /*
     * Initialize write cursors to track progress across batches for each src
     *   - recv_displs_src[src] = base offset in recv_buffer for each src (computed once)
     *   - src_write_cursor[src] = current write position within each src's region (starts at 0)
     *   - Each batch, displacement = recv_displs_src[src] + src_write_cursor[src]
     *   - After receiving data, cursor advances: src_write_cursor[src] += recvcounts_batch[src]
     *
     * This allows multiple batches to write sequentially into recv_buffer
     * w/o overwriting. Cursor "shifts" through each src's allocated region
     * as batches are processed, so that data from batch N is written after data from
     * batches 0 to N-1 for that src.
     */

    src_write_cursor = (int64_t*)calloc(num_ranks, sizeof(int64_t));
    
    
    // Build Y --> (SRC, LOCAL_IDX) mapping
    y_owner_src = (int*)malloc(sizeof(int) * N);
    y_src_local_idx = (int*)malloc(sizeof(int) * N);
    y_batch_idx = (int*)malloc(sizeof(int) * N);
    y_slice_idx_in_batch = (int*)malloc(sizeof(int) * N);
    
    // Initialize 
    for (int y = 0; y < N; y++) {
        y_owner_src[y] = -1;
        y_src_local_idx[y] = -1;
        y_batch_idx[y] = -1;
        y_slice_idx_in_batch[y] = -1;
    }
    
    // Allocate src_batch_slice_counts to track slices per batch per source
    src_batch_slice_counts = (int**)malloc(sizeof(int*) * num_ranks);
    for (int src = 0; src < num_ranks; src++) {
        src_batch_slice_counts[src] = (int*)calloc(global_max_batches, sizeof(int));
    }
    
    // Build mapping by simulating what each source generates
    int *src_y_counter = (int*)calloc(num_ranks, sizeof(int));
    
    for (int src = 0; src < num_ranks; src++) {
        for (int batch = 0; batch < global_max_batches; batch++) {
            int y_values[2];
            int count;
            get_rank_batch_y_values(src, batch, N, num_ranks, y_values, &count);
            src_batch_slice_counts[src][batch] = count;
            for (int i = 0; i < count; i++) {
                int y_global = y_values[i];
                if (y_owner_src[y_global] != -1) {
                    fprintf(stderr, "[ERROR] Y=%d already assigned to src %d, now src %d wants it!\n",
                           y_global, y_owner_src[y_global], src);
                    MPI_Abort(zd_comm_2d, 1);
                }
                
                y_owner_src[y_global] = src;
                y_src_local_idx[y_global] = src_y_counter[src];
                y_batch_idx[y_global] = batch;
                y_slice_idx_in_batch[y_global] = i;
                src_y_counter[src]++;
            }
        }
    }
    
    free(src_y_counter);

    // ===== Allocate persistent thread-local RNG buffers =====
    // (Commented out: ZD_MPI_generation now allocates per-call inside parallel for)
    void** thread_rng_buffers = NULL;
// #if PARALLELIZE_Z_LOOP
//     if (!is_idle_rank) {
//         int max_threads = omp_get_max_threads();
//         thread_rng_buffers = (void**)malloc(sizeof(void*) * max_threads);
//         size_t rng_size = zeldovich_ps_rng_buffer_size();
//         for (int t = 0; t < max_threads; t++) {
//             thread_rng_buffers[t] = malloc(rng_size);
//         }
//         if (rank == 0) {
//             printf("[PERFORMANCE] Allocated %d persistent RNG buffers of %zu bytes each\n",
//                    max_threads, rng_size);
//         }
//     }
// #endif
    // ========================================================================
    // STAGE 5: MAIN MULTI-BATCH LOOP
    // ========================================================================
    // GENERATION + COMMUNICATION + ACCUMULATION
    if (rank == 0) {
        printf("\n[MULTI-BATCH] Starting batch processing...\n");
        printf("              Ranks will process %d pairs each (approx)\n",
               is_idle_rank ? 0 : my_num_pairs);
        printf("              sizeof(MPI_Aint)=%zu (%s), sizeof(MPI_Count)=%zu\n",
               sizeof(MPI_Aint), sizeof(MPI_Aint) == 8 ? "int64" : (sizeof(MPI_Aint) == 4 ? "int32" : "other"),
               sizeof(MPI_Count));
        fflush(stdout);
    }

    STimer t_gen, t_comm;
    t_gen.Start();
    
    for (int batch_idx = 0; batch_idx < global_max_batches; batch_idx++) {
        // ===== BATCH STEP 1: Get this batch's data =====
        int my_batch_slice_count = 0;
        int y_batch_primary = -1, y_batch_mirror = -1;
        bool batch_is_self_conjugate = false;
        
        if (!is_idle_rank && batch_idx < my_num_pairs) {
            y_batch_primary = my_pair_list[batch_idx].y_primary;
            y_batch_mirror = my_pair_list[batch_idx].y_mirror;
            batch_is_self_conjugate = my_pair_list[batch_idx].is_self_conjugate;
            my_batch_slice_count = batch_is_self_conjugate ? 1 : 2;
        }
        
        // ===== BATCH STEP 2: Generate + 2D FFT =====
        if (!is_idle_rank && my_batch_slice_count > 0) {
            if (posix_memalign((void**)&local_y_slices, ALIGN_BYTES, local_y_slices_bytes) != 0) {
                fprintf(stderr, "Rank %d: posix_memalign failed for local_y_slices (batch %d)\n",
                        rank, batch_idx);
                MPI_Abort(zd_comm_2d, 1);
            }
            if (posix_memalign((void**)&fft_stage_2d, ALIGN_BYTES, fft_stage_bytes) != 0) {
                fprintf(stderr, "Rank %d: posix_memalign failed for fft_stage_2d (batch %d)\n",
                        rank, batch_idx);
                MPI_Abort(zd_comm_2d, 1);
            }

            memset(local_y_slices, 0, local_y_slices_bytes);

            // Generate this batch's pair
            // For self-conjugate slices, conjugate_slices acts as temporary storage
            // Use int64_t index to avoid overflow: narray*N*N can exceed INT_MAX for N > ~23170
            int64_t slice_elems = (int64_t)narray * (int64_t)N * (int64_t)N;
            fftw_complex_t *primary_ptr   = &local_y_slices[0];
            fftw_complex_t *conjugate_ptr = &local_y_slices[slice_elems];
            
            generate_zd_mpi_slice_pair_local(
                N, y_batch_primary, y_batch_mirror,
                primary_ptr, conjugate_ptr, //primary_ptr is D+i*F
                narray, plan_2d, fft_stage_2d, rank,
                ps, params, ps_primordial,
                thread_rng_buffers
            );

            free(fft_stage_2d);
            fft_stage_2d = NULL;
        }
        
        // ===== BATCH STEP 3: Calculate send/recv counts =====
        int64_t *sendcounts_batch, *sdispls_batch, *recvcounts_batch;
        int64_t total_send_batch, total_recv_batch;

        calculate_batch_send_recv_counts(
            rank, num_ranks, N, narray, batch_idx,
            my_batch_slice_count, my_pencils,
            grid_x, grid_z, cpd,
            &sendcounts_batch, &sdispls_batch,
            &recvcounts_batch,
            &total_send_batch, &total_recv_batch
        );
        
        std::vector<int64_t> rdispls_elem(num_ranks);
        for (int src = 0; src < num_ranks; src++) {
            int64_t new_displ = recv_displs_src[src] + src_write_cursor[src];
            if (new_displ < 0) {
                fprintf(stderr, "[Rank %d] ERROR: Negative rdispls! src=%d, displ=%lld\n",
                       rank, src, (long long)new_displ);
                MPI_Abort(zd_comm_2d, 1);
            }
            rdispls_elem[src] = new_displ;
        }
        
        // ===== BATCH STEP 4: Allocate per-batch send buffer =====
        fftw_complex_t *send_buffer_batch = NULL;
        
        if (total_send_batch > 0) {
            size_t requested_bytes = (size_t)total_send_batch * sizeof(fftw_complex_t);
            size_t alloc_bytes = ROUND_UP_PAGE(requested_bytes);
            if (posix_memalign((void**)&send_buffer_batch, ALIGN_BYTES, alloc_bytes) != 0) {
                fprintf(stderr, "Rank %d: posix_memalign failed for send_buffer_batch\n", rank);
                MPI_Abort(zd_comm_2d, 1);
            }
        }
        
        // ===== BATCH STEP 5: Pack to send buffer=====
        if (!is_idle_rank && my_batch_slice_count > 0 && total_send_batch > 0) {
            pack_slices_to_send_buffer(
                rank, num_ranks, N, narray,
                local_y_slices, my_batch_slice_count, NULL,
                send_buffer_batch, sendcounts_batch, sdispls_batch,
                grid_x, grid_z, cpd
            );
        }

        // Free local_y_slices before MPI so peak comm memory is recv_buffer + send_buffer only.
        if (local_y_slices != NULL) {
            free(local_y_slices);
            local_y_slices = NULL;
        }
        
        // ===== BATCH STEP 6: MPI_Alltoallv_c =====
        #if DEBUG_PRINTS
        // Verify sendcounts sum matches total_send_batch
        if (send_buffer_batch != NULL && total_send_batch > 0) {
            int64_t sum_sendcounts = 0;
            for (int i = 0; i < num_ranks; i++) {
                sum_sendcounts += sendcounts_batch[i];
            }
            if (sum_sendcounts != total_send_batch) {
                fprintf(stderr, "[Rank %d] ERROR: sendcounts sum (%lld) != total_send_batch (%lld)!\n",
                       rank, (long long)sum_sendcounts, (long long)total_send_batch);
                MPI_Abort(zd_comm_2d, 1);
            }
        }
        
        // Verify recvcounts sum matches
        if (recv_buffer != NULL) {
            int64_t sum_recvcounts = 0;
            for (int i = 0; i < num_ranks; i++) {
                sum_recvcounts += recvcounts_batch[i];
            }
            if (sum_recvcounts != total_recv_batch) {
                fprintf(stderr, "[Rank %d] ERROR: recvcounts sum (%lld) != total_recv_batch (%lld)!\n",
                       rank, (long long)sum_recvcounts, (long long)total_recv_batch);
                MPI_Abort(zd_comm_2d, 1);
            }
            
            // Verify rdispls don't exceed recv_buffer bounds
            int64_t recv_buffer_size = recv_total_elems;
            for (int i = 0; i < num_ranks; i++) {
                if (recvcounts_batch[i] > 0) {
                    int64_t end_offset = rdispls_elem[i] + recvcounts_batch[i];
                    if (end_offset > recv_buffer_size) {
                        fprintf(stderr, "[Rank %d] ERROR: rdispls[%d] + recvcounts[%d] = %ld exceeds buffer size %ld!\n",
                               rank, i, i, (long)end_offset, (long)recv_buffer_size);
                        MPI_Abort(zd_comm_2d, 1);
                    }
                }
            }
        }
        {
            int64_t max_send_displ = 0, max_recv_displ = 0;
            for (int i = 0; i < num_ranks; i++) {
                int64_t send_end = (int64_t)sdispls_batch[i] + sendcounts_batch[i];
                int64_t recv_end = rdispls_elem[i] + recvcounts_batch[i];
                
                if (send_end > max_send_displ) max_send_displ = send_end;
                if (recv_end > max_recv_displ) max_recv_displ = recv_end;
                
                if (sdispls_batch[i] < 0 || rdispls_elem[i] < 0) {
                    fprintf(stderr, "[Rank %d] ERROR: Negative displacement! sdispls[%d]=%lld, rdispls[%d]=%lld\n",
                           rank, i, (long long)sdispls_batch[i], i, (long long)rdispls_elem[i]);
                    MPI_Abort(zd_comm_2d, 1);
                }
            }
            
            if (max_send_displ > total_send_batch) {
                fprintf(stderr, "[Rank %d] ERROR: Send displacement exceeds buffer! max=%lld > total=%lld\n",
                       rank, (long long)max_send_displ, (long long)total_send_batch);
                MPI_Abort(zd_comm_2d, 1);
            }
            
            if (max_recv_displ > recv_total_elems) {
                fprintf(stderr, "[Rank %d] ERROR: Recv displacement exceeds buffer! max=%lld > total=%lld\n",
                       rank, (long long)max_recv_displ, (long long)recv_total_elems);
                MPI_Abort(zd_comm_2d, 1);
            }
        }
        #endif
        
        if (batch_idx == 0 && rank == 0) {
            long long max_send = 0, max_recv = 0;
            for (int i = 0; i < num_ranks; i++) {
                if ((long long)sendcounts_batch[i] > max_send) max_send = (long long)sendcounts_batch[i];
                if ((long long)recvcounts_batch[i] > max_recv) max_recv = (long long)recvcounts_batch[i];
            }
            fprintf(stdout, "[MPI-DIAG] Batch 0: max_send=%lld max_recv=%lld elems (%.3f GB / %.3f GB)\n",
                    max_send, max_recv,
                    max_send * sizeof(fftw_complex_t) / 1.0e9,
                    max_recv * sizeof(fftw_complex_t) / 1.0e9);
            fflush(stdout);
        }

        t_comm.Start();
        {
            /* Counts/displacements stay int64 in element units (not bytes): MPI_Alltoallv_c on
             * MPI >= 4, Isend/Irecv fallback otherwise. Receive displacements reach
             * ~4*N^3/num_ranks, which overflows the int arguments of plain MPI_Alltoallv. */
            int mpi_err = zd_alltoallv_int64(
                send_buffer_batch, sendcounts_batch, sdispls_batch,
                recv_buffer, recvcounts_batch, rdispls_elem.data(),
                MPI_COMPLEX_TYPE, num_ranks, zd_comm_2d, 0
            );
            if (mpi_err != MPI_SUCCESS) {
                char errstr[MPI_MAX_ERROR_STRING];
                int errlen;
                MPI_Error_string(mpi_err, errstr, &errlen);
                fprintf(stderr, "[Rank %d] Alltoallv FAILED (batch %d): %s\n",
                        rank, batch_idx, errstr);
                for (int i = 0; i < num_ranks; i++) {
                    fprintf(stderr, "  send[%d]=%lld sdisp[%d]=%lld recv[%d]=%lld rdisp[%d]=%lld\n",
                            i, (long long)sendcounts_batch[i], i, (long long)sdispls_batch[i],
                            i, (long long)recvcounts_batch[i], i, (long long)rdispls_elem[i]);
                }
                fflush(stderr);
                MPI_Abort(zd_comm_2d, mpi_err);
            }
        }
        t_comm.Stop();
        
        // ===== BATCH STEP 7: Update write cursors =====
        // After MPI_Alltoallv_c completes, the cursor is advanced by recvcounts_batch[src]
        // The next batch then calculates a displacement that points after the current batch's data
        // (loop over batches of y slices) Each MPI_Alltoallv_c only adds the next chunk of data into the right place in the same pre-sized recv_buffer. The cursors (src_write_cursor) make batch 1’s data sit after batch 0’s data for each source, and so on, until the buffer matches the final “source-grouped” layout described in the file header.

        for (int src = 0; src < num_ranks; src++) {
            src_write_cursor[src] += recvcounts_batch[src];
        }
        
        // ===== BATCH STEP 8: Free per-batch buffers =====
        if (send_buffer_batch) free(send_buffer_batch);
        free(sendcounts_batch);
        free(sdispls_batch);
        free(recvcounts_batch);
        
        // Progress indicator
        if (rank == 0 && global_max_batches > 5) {
            if ((batch_idx + 1) % (global_max_batches / 10 + 1) == 0 || batch_idx == global_max_batches - 1) {
                printf("         ... Completed %d/%d batches (%.1f%%)\n",
                       batch_idx + 1, global_max_batches, 
                       100.0 * (batch_idx + 1) / global_max_batches);
            }
        }
    }
    
    t_gen.Stop();
    
    print_zd_mpi_gen_timers(rank);
    
    if (rank == 0) {
        printf("[MULTI-BATCH] All batches complete.\n");
        printf("              Generation time: %.6f s\n", t_gen.Elapsed());
        printf("              Communication time: %.6f s\n", t_comm.Elapsed());
    }
    
    if (local_y_slices != NULL) {
        fprintf(stderr, "[WARNING] Rank %d: local_y_slices still allocated after batch loop\n", rank);
        free(local_y_slices);
        local_y_slices = NULL;
    }
    // fft_stage_2d is per-batch (freed after each generate+2D FFT); safety check:
    if (fft_stage_2d != NULL) {
        fprintf(stderr, "[WARNING] Rank %d: fft_stage_2d still allocated after batch loop\n", rank);
        free(fft_stage_2d);
        fft_stage_2d = NULL;
    }
    
    // ========================================================================
    // V11: VERIFY CURSORS AFTER ALL BATCHES
    // ========================================================================
    
    if (!is_idle_rank) {
        bool cursor_error = false;
        for (int src = 0; src < num_ranks; src++) {
            int64_t expected = (int64_t)src_total_slices[src] * my_pencils * narray;
            if (src_write_cursor[src] != expected) {
                fprintf(stderr, "[ERROR] Rank %d: src %d cursor mismatch! Got %lld, expected %lld\n",
                       rank, src, (long long)src_write_cursor[src], (long long)expected);
                cursor_error = true;
            }
        }
        
        if (cursor_error) {
            MPI_Abort(zd_comm_2d, 1);
        }
    }
    
    #if !FREE_PCG_AFTER_STAGE1
    cleanup_global_pcg();
    #endif
    
    // ========================================================================
    // V12: STAGE 3 - Z-SLAB STREAMING (FROM SOURCE-GROUPED RECV_BUFFER)
    // ========================================================================
    
    if (rank == 0) {
        printf("\n[Stage 3] Z-slab streaming from recv_buffer...\n");
    }
    
    // --- Z-SLAB STREAMING (one Z-slab at a time, [Array][X][Y] then transpose to [Array][Y][X]) ---
    STimer t_streaming;
    t_streaming.Start();
    
    if (rank == 0) {
        printf("\n[Stage 3] Z-slab streaming: Unpack --> FFT --> Write (Zeldovich format)...\n");
    }
    
    // V12: Allocate local_z_slab for ONE Z-SLAB ONLY
    // V13: Use appropriate bounds for allocation (padded if enabled, core otherwise)
    int64_t elements_per_z_slab = 0;
    if (!is_idle_rank) {
#if USE_X_PADDING
        int x_count = my_extended_bounds.padded.x_end - my_extended_bounds.padded.x_start;
#else
        int x_count = my_extended_bounds.core.x_end - my_extended_bounds.core.x_start;
#endif
        // Allocate for [Array][X][Y] format: narray * x_count * N (Y stride-1 for FFT)
        elements_per_z_slab = (int64_t)narray * x_count * N;
        
        if (posix_memalign((void**)&local_z_slab, ALIGN_BYTES, 
                           sizeof(fftw_complex_t) * elements_per_z_slab) != 0) {
            fprintf(stderr, "Rank %d: posix_memalign failed for local_z_slab (one Z-slab)\n", rank);
            MPI_Abort(zd_comm_2d, 1);
        }

    } else {
        // Idle ranks: local_z_slab stays NULL
        local_z_slab = NULL;
    }
    
    // Create directory for this rank (before Z-loop) -- only needed for Mode 1 or 2 (.bin files)
#if (PARTICLE_OUTPUT_MODE != 3 && PARTICLE_OUTPUT_MODE != 4)
    char dirname[64];
    snprintf(dirname, sizeof(dirname), "rank_%d", rank);
    int mkdir_result = mkdir(dirname, 0755);
    if (mkdir_result != 0 && errno != EEXIST) {
        fprintf(stderr, "Rank %d: ERROR creating directory %s (errno=%d)\n", rank, dirname, errno);
        MPI_Abort(zd_comm_2d, 1);
    }
#endif
    // Process one Z-slab at a time 
    int files_written = 0;
    size_t total_bytes_written = 0;
    
    // MODE 3: file vectors for x-slab format (grid_x>1) or z-group format (grid_x==1)
    int slab_x_start = 0, slab_x_end = 0;
    std::vector<FILE*> slab_fp;
    std::vector<FILE*> slab_dens_fp;

    int zgrp_start = 0, zgrp_end = 0;
    std::vector<FILE*> zgrp_fp;
    std::vector<FILE*> zgrp_dens_fp;

    // MODE 4: file vectors for z-slab format (paths use z%%03d / ic_*_z%%03d; rank index is still rank_x)
    int slab_z_start = 0, slab_z_end = 0;
    std::vector<FILE*> zslab_fp;
    std::vector<FILE*> zslab_dens_fp;
    
    if (PARTICLE_OUTPUT_MODE == 3 && params != NULL && !is_idle_rank) {
        ZeldovichParameters *p = static_cast<ZeldovichParameters*>(params);

        if (grid_x == 1) {
            // ---------------------------------------------------------------
            // grid_x==1: z-group file mapping (Abacus RVZel_2D, MPI_size_z==1)
            // Each z-rank writes CPD/grid_z files: ic_{file_index:04d}
            // file_index = z * CPD / N, each file holds N/CPD z-planes.
            // ---------------------------------------------------------------
            // old code:
            // int s_z_start = (rank_z * cpd) / grid_z;
            // int s_z_end   = ((rank_z + 1) * cpd) / grid_z;

            // Slab band follows rank_x (Abacus x-decomposition), NOT rank_z.
            // After the cart transpose (dims={grid_z, grid_x}), grid_x==1 pins rank_z=0
            // for all ranks, so rank_x is the varying coordinate that selects the slab band.
            // Matches the flat Abacus reader (MPI_size_z==1): InitialConditionsDirectory/ic_%04d.
            int s_z_start = (rank_x * cpd) / grid_z;
            int s_z_end   = ((rank_x + 1) * cpd) / grid_z;
            zgrp_start = s_z_start;
            zgrp_end   = s_z_end;

            zgrp_fp.resize(zgrp_end - zgrp_start, NULL);
            zgrp_dens_fp.resize(zgrp_end - zgrp_start, NULL);

            for (int f = zgrp_start; f < zgrp_end; f++) {
                char fp_path[PATH_MAX];
                snprintf(fp_path, sizeof(fp_path), "%s/ic_%04d",
                         p->output_dir.c_str(), f);
                zgrp_fp[f - zgrp_start] = fopen(fp_path, "wb");
                if (!zgrp_fp[f - zgrp_start]) {
                    fprintf(stderr, "Rank %d: ERROR opening %s for writing (errno=%d)\n",
                            rank, fp_path, errno);
                }
                if (p->qdensity && zgrp_fp[f - zgrp_start] != NULL) {
                    char fd_path[PATH_MAX];
                    snprintf(fd_path, sizeof(fd_path), "%s/dens_%04d",
                             p->output_dir.c_str(), f);
                    zgrp_dens_fp[f - zgrp_start] = fopen(fd_path, "wb");
                    if (!zgrp_dens_fp[f - zgrp_start]) {
                        fprintf(stderr, "Rank %d: ERROR opening %s for density (errno=%d)\n",
                                rank, fd_path, errno);
                    }
                }
            }

            if (rank == 0) {
                printf("[MODE 3] grid_x==1, z-group format (RVZel_2D flat): cpd=%d, files_per_rank=%d, ic_%%04d\n",
                       cpd, zgrp_end - zgrp_start);
                printf("         Each file holds %d z-planes of %d x %d particles\n",
                       N / cpd, N, N);
            }
        } else {
            // ---------------------------------------------------------------
            // grid_x>1: x-slab file mapping (existing Mode 3)
            // Standalone: {output_dir}/ic/z###/... ; embedded Abacus: {output_dir}/z###/...
            // ---------------------------------------------------------------
            if (!zeldovich_ic_embedded) {
                char ic_dir[PATH_MAX];
                snprintf(ic_dir, sizeof(ic_dir), "%s/ic", p->output_dir.c_str());
                int mkdir_ic = mkdir(ic_dir, 0755);
                if (mkdir_ic != 0 && errno != EEXIST) {
                    fprintf(stderr, "Rank %d: ERROR creating directory %s (errno=%d)\n", rank, ic_dir, errno);
                    MPI_Abort(zd_comm_2d, 1);
                }
            }
            if (p->qdensity) {
                char dens_dir[PATH_MAX];
                snprintf(dens_dir, sizeof(dens_dir), "%s/dens", p->output_dir.c_str());
                int mkdir_dens = mkdir(dens_dir, 0755);
                if (mkdir_dens != 0 && errno != EEXIST) {
                    fprintf(stderr, "Rank %d: ERROR creating directory %s (errno=%d)\n", rank, dens_dir, errno);
                    MPI_Abort(zd_comm_2d, 1);
                }
            }

            slab_x_start = (rank_x * cpd) / grid_x;
            slab_x_end   = ((rank_x + 1) * cpd) / grid_x;

            char z_dir[PATH_MAX];
            z_dir[0] = '\0';
            if (grid_z > 1) {
                if (zeldovich_ic_embedded) {
                    snprintf(z_dir, sizeof(z_dir), "%s/z%03d", p->output_dir.c_str(), rank_z);
                } else {
                    snprintf(z_dir, sizeof(z_dir), "%s/ic/z%03d", p->output_dir.c_str(), rank_z);
                }
                int mkdir_z = mkdir(z_dir, 0755);
                if (mkdir_z != 0 && errno != EEXIST) {
                    fprintf(stderr, "Rank %d: ERROR creating directory %s (errno=%d)\n", rank, z_dir, errno);
                    MPI_Abort(zd_comm_2d, 1);
                }
                if (p->qdensity) {
                    char dens_z_dir[PATH_MAX];
                    snprintf(dens_z_dir, sizeof(dens_z_dir), "%s/dens/z%03d", p->output_dir.c_str(), rank_z);
                    int mkdir_dens_z = mkdir(dens_z_dir, 0755);
                    if (mkdir_dens_z != 0 && errno != EEXIST) {
                        fprintf(stderr, "Rank %d: ERROR creating directory %s (errno=%d)\n", rank, dens_z_dir, errno);
                        MPI_Abort(zd_comm_2d, 1);
                    }
                }
            }

            slab_fp.resize(slab_x_end - slab_x_start, NULL);
            slab_dens_fp.resize(slab_x_end - slab_x_start, NULL);

            for (int s = slab_x_start; s < slab_x_end; s++) {
                char fp_path[PATH_MAX];
                if (grid_z > 1) {
                    if (zeldovich_ic_embedded) {
                        snprintf(fp_path, sizeof(fp_path), "%s/z%03d/ic_%04d_z%03d",
                                 p->output_dir.c_str(), rank_z, s, rank_z);
                    } else {
                        snprintf(fp_path, sizeof(fp_path), "%s/ic/z%03d/ic_%04d_z%03d",
                                 p->output_dir.c_str(), rank_z, s, rank_z);
                    }
                } else {
                    if (zeldovich_ic_embedded) {
                        snprintf(fp_path, sizeof(fp_path), "%s/ic_%04d",
                                 p->output_dir.c_str(), s);
                    } else {
                        snprintf(fp_path, sizeof(fp_path), "%s/ic/ic_%04d",
                                 p->output_dir.c_str(), s);
                    }
                }
                slab_fp[s - slab_x_start] = fopen(fp_path, "wb");
                if (!slab_fp[s - slab_x_start]) {
                    fprintf(stderr, "Rank %d: ERROR opening %s for writing (errno=%d)\n",
                            rank, fp_path, errno);
                }
                if (p->qdensity && slab_fp[s - slab_x_start] != NULL) {
                    char fd_path[PATH_MAX];
                    if (grid_z > 1) {
                        snprintf(fd_path, sizeof(fd_path), "%s/dens/z%03d/dens_%04d",
                                 p->output_dir.c_str(), rank_z, s);
                    } else {
                        snprintf(fd_path, sizeof(fd_path), "%s/dens/dens_%04d",
                                 p->output_dir.c_str(), s);
                    }
                    slab_dens_fp[s - slab_x_start] = fopen(fd_path, "wb");
                    if (!slab_dens_fp[s - slab_x_start]) {
                        fprintf(stderr, "Rank %d: ERROR opening %s for density (errno=%d)\n",
                                rank, fd_path, errno);
                    }
                }
            }

            if (!is_idle_rank && grid_x > 1) {
                fprintf(stderr,
                        "[IC_WRITE_DEBUG] MPI_rank=%d (rank_x=%d, rank_z=%d) MODE3 writer ic band "
                        "file_index in [%d,%d) (%d files) output_dir=%s\n",
                        rank, rank_x, rank_z, slab_x_start, slab_x_end,
                        slab_x_end - slab_x_start, p->output_dir.c_str());
                if (slab_x_end > slab_x_start) {
                    fprintf(stderr,
                            "[IC_WRITE_DEBUG] MPI_rank=%d writes ic_%04d .. ic_%04d\n",
                            rank, slab_x_start, slab_x_end - 1);
                }
                fflush(stderr);
            }

            if (rank == 0) {
                if (grid_z > 1) {
                    if (zeldovich_ic_embedded) {
                        printf("[MODE 3] One file per x-slab and z-rank: cpd=%d, slabs_per_rank=%d, z%%03d/ic_%%04d_z%%03d (embedded)\n",
                               cpd, slab_x_end - slab_x_start);
                    } else {
                        printf("[MODE 3] One file per x-slab and z-rank: cpd=%d, slabs_per_rank=%d, ic/z%%03d/ic_%%04d_z%%03d\n",
                               cpd, slab_x_end - slab_x_start);
                    }
                } else {
                    if (zeldovich_ic_embedded) {
                        printf("[MODE 3] One file per x-slab: cpd=%d, slabs_per_rank=%d, ic_%%04d (embedded)\n",
                               cpd, slab_x_end - slab_x_start);
                    } else {
                        printf("[MODE 3] One file per x-slab: cpd=%d, slabs_per_rank=%d, ic/ic_%%04d\n",
                               cpd, slab_x_end - slab_x_start);
                    }
                }
            }
        }
    }

    // MODE 4: z(k)-slab files under ic/z%03d/ standalone, or z%03d/ when embedded in Abacus
    if (PARTICLE_OUTPUT_MODE == 4 && params != NULL && !is_idle_rank) {
        ZeldovichParameters *p = static_cast<ZeldovichParameters*>(params);

        if (!zeldovich_ic_embedded) {
            char ic_dir[PATH_MAX];
            snprintf(ic_dir, sizeof(ic_dir), "%s/ic", p->output_dir.c_str());
            int mkdir_ic = mkdir(ic_dir, 0755);
            if (mkdir_ic != 0 && errno != EEXIST) {
                fprintf(stderr, "Rank %d: ERROR creating directory %s (errno=%d)\n", rank, ic_dir, errno);
                MPI_Abort(zd_comm_2d, 1);
            }
        }

        // z%03d/ per rank_x (Abacus "x" split); z_ prefix is on-disk convention only
        char ic_z_subdir[PATH_MAX];
        // old code:
        // if (zeldovich_ic_embedded) {
        //     snprintf(ic_z_subdir, sizeof(ic_z_subdir), "%s/z%03d", p->output_dir.c_str(), rank_x);
        // } else {
        //     snprintf(ic_z_subdir, sizeof(ic_z_subdir), "%s/ic/z%03d", p->output_dir.c_str(), rank_x);
        // }

        // new change:
        if (zeldovich_ic_embedded) {
            snprintf(ic_z_subdir, sizeof(ic_z_subdir), "%s/z%03d", p->output_dir.c_str(), rank_z);
        } else {
            snprintf(ic_z_subdir, sizeof(ic_z_subdir), "%s/ic/z%03d", p->output_dir.c_str(), rank_z);
        }
        int mkdir_ic_z = mkdir(ic_z_subdir, 0755);
        if (mkdir_ic_z != 0 && errno != EEXIST) {
            fprintf(stderr, "Rank %d: ERROR creating directory %s (errno=%d)\n", rank, ic_z_subdir, errno);
            MPI_Abort(zd_comm_2d, 1);
        }

        if (p->qdensity) {
            char dens_dir[PATH_MAX];
            snprintf(dens_dir, sizeof(dens_dir), "%s/dens", p->output_dir.c_str());
            int mkdir_dens = mkdir(dens_dir, 0755);
            if (mkdir_dens != 0 && errno != EEXIST) {
                fprintf(stderr, "Rank %d: ERROR creating directory %s (errno=%d)\n", rank, dens_dir, errno);
                MPI_Abort(zd_comm_2d, 1);
            }
            char dens_z_subdir[PATH_MAX];
            // old code:
            // snprintf(dens_z_subdir, sizeof(dens_z_subdir), "%s/dens/z%03d", p->output_dir.c_str(), rank_x);
            // new change:
            snprintf(dens_z_subdir, sizeof(dens_z_subdir), "%s/dens/z%03d", p->output_dir.c_str(), rank_z);
            int mkdir_dens_z = mkdir(dens_z_subdir, 0755);
            if (mkdir_dens_z != 0 && errno != EEXIST) {
                fprintf(stderr, "Rank %d: ERROR creating directory %s (errno=%d)\n", rank, dens_z_subdir, errno);
                MPI_Abort(zd_comm_2d, 1);
            }
        }

        // old code:
        // // Z(k)-slab ownership: partition cpd z-slabs among z(k)-ranks
        // slab_z_start = (rank_z * cpd) / grid_z;
        // slab_z_end   = ((rank_z + 1) * cpd) / grid_z;

        // new change:
        // Slab band follows rank_x (Abacus x-decomposition), NOT rank_z.
        // rank_x = coords[0] ranges over dims[0] = grid_z (= num_ranks/NumZRanks = MPI_size_x).
        slab_z_start = (rank_x * cpd) / grid_z;
        slab_z_end   = ((rank_x + 1) * cpd) / grid_z;

        zslab_fp.resize(slab_z_end - slab_z_start, NULL);
        zslab_dens_fp.resize(slab_z_end - slab_z_start, NULL);

        // Zeldovich-MPI "z" = Abacus "x" T^T
        for (int s = slab_z_start; s < slab_z_end; s++) {
            char fp_path[PATH_MAX];
            if (zeldovich_ic_embedded) {
                // old code:
                // snprintf(fp_path, sizeof(fp_path), "%s/z%03d/ic_%04d_z%03d",
                //          p->output_dir.c_str(), rank_x, s, rank_x);
                // new change:
                snprintf(fp_path, sizeof(fp_path), "%s/z%03d/ic_%04d_z%03d", p->output_dir.c_str(), rank_z, s, rank_z);
            } else {
                // old code:
                // snprintf(fp_path, sizeof(fp_path), "%s/ic/z%03d/ic_%04d_z%03d",
                //          p->output_dir.c_str(), rank_x, s, rank_x);
                // new change:
                snprintf(fp_path, sizeof(fp_path), "%s/ic/z%03d/ic_%04d_z%03d",
                         p->output_dir.c_str(), rank_z, s, rank_z);
            }
            zslab_fp[s - slab_z_start] = fopen(fp_path, "wb");
            if (!zslab_fp[s - slab_z_start]) {
                fprintf(stderr, "Rank %d: ERROR opening %s for writing (errno=%d)\n",
                        rank, fp_path, errno);
            }
            if (p->qdensity && zslab_fp[s - slab_z_start] != NULL) {
                char fd_path[PATH_MAX];
                // old code:
                // snprintf(fd_path, sizeof(fd_path), "%s/dens/z%03d/dens_%04d",
                //          p->output_dir.c_str(), rank_x, s);
                // new change:
                snprintf(fd_path, sizeof(fd_path), "%s/dens/z%03d/dens_%04d",
                         p->output_dir.c_str(), rank_z, s);
                zslab_dens_fp[s - slab_z_start] = fopen(fd_path, "wb");
                if (!zslab_dens_fp[s - slab_z_start]) {
                    fprintf(stderr, "Rank %d: ERROR opening %s for density (errno=%d)\n",
                            rank, fd_path, errno);
                }
            }
        }

        // old code:
        // fprintf(stderr,
        //     "[IC_WRITE_DEBUG] Zeldovich writer: world_rank=%d comm_2d_rank=%d "
        //     "cart(rank_x=%d, rank_z=%d) embedded=%d -> %s/ic_%%04d_z%03d "
        //     "slabs [%d,%d) (%d files)\n",
        //     world_rank, rank, rank_x, rank_z, zeldovich_ic_embedded ? 1 : 0,
        //     ic_z_subdir, rank_x,
        //     slab_z_start, slab_z_end, slab_z_end - slab_z_start);
        // fflush(stderr);

        // new change:
        // fprintf(stderr,
        //     "[IC_WRITE_DEBUG] Zeldovich writer: world_rank=%d comm_2d_rank=%d "
        //     "cart(rank_x=%d, rank_z=%d) embedded=%d -> %s/ic_%%04d_z%03d "
        //     "slabs [%d,%d) (%d files)\n",
        //     world_rank, rank, rank_x, rank_z, zeldovich_ic_embedded ? 1 : 0,
        //     ic_z_subdir, rank_z,
        //     slab_z_start, slab_z_end, slab_z_end - slab_z_start);
        // fflush(stderr);

        if (rank == 0) {
            if (zeldovich_ic_embedded) {
                printf("[MODE 4] z-slab files: cpd=%d, slabs_per_rank=%d, z%%03d/ic_%%04d_z%%03d (embedded; dens/z%%03d/dens_%%04d if qdensity)\n",
                       cpd, slab_z_end - slab_z_start);
            } else {
                printf("[MODE 4] z-slab files: cpd=%d, slabs_per_rank=%d, ic/z%%03d/ic_%%04d_z%%03d (and dens/z%%03d/dens_%%04d if qdensity)\n",
                       cpd, slab_z_end - slab_z_start);
            }
        }
    }

    double acc_unpack = 0.0;
    double acc_fft = 0.0;
    double acc_io = 0.0;
    
    if (!is_idle_rank && my_pencils > 0) {
        // allocate array of pointers, one per thread
        // pointers to aligned buffers for 1D Y FFT
        const int num_threads = omp_get_max_threads();
        fftw_complex_t **thread_1d_bufs =
            (fftw_complex_t**)calloc((size_t)num_threads, sizeof(fftw_complex_t*));
        if (thread_1d_bufs == NULL) {
            fprintf(stderr, "Rank %d: calloc failed for thread_1d_bufs\n", rank);
            MPI_Abort(zd_comm_2d, 1);
        }
        // each thread gets aligned buffer of size PPD for 1D FFT
        for (int t = 0; t < num_threads; t++) {
            if (posix_memalign((void**)&thread_1d_bufs[t], ALIGN_BYTES,
                               sizeof(fftw_complex_t) * (size_t)N) != 0) {
                fprintf(stderr, "Rank %d: posix_memalign failed for thread_1d_bufs[%d]\n", rank, t);
                MPI_Abort(zd_comm_2d, 1);
            }
        }
        const int num_thread_bufs = num_threads;

        // Use appropriate bounds for Z-loop (padded if enabled, core otherwise)
#if USE_X_PADDING
        int x_count = my_extended_bounds.padded.x_end - my_extended_bounds.padded.x_start;
        int z_count = my_extended_bounds.padded.z_end - my_extended_bounds.padded.z_start;
        
        for (int z = my_extended_bounds.padded.z_start; z < my_extended_bounds.padded.z_end; z++) {
#else
        int x_count = my_extended_bounds.core.x_end - my_extended_bounds.core.x_start;
        int z_count = my_extended_bounds.core.z_end - my_extended_bounds.core.z_start;
        
        for (int z = my_extended_bounds.core.z_start; z < my_extended_bounds.core.z_end; z++) {
#endif
            // Clear buffer for this Z-slab
            memset(local_z_slab, 0, sizeof(fftw_complex_t) * elements_per_z_slab);
            
            // V12: Unpack + FFT this Z-slab from recv_buffer (source-grouped)
            // V14+: Fixed batch-aware unpacking formula
            z_streaming_unpack(
                rank, N, narray,
                z,                               // Current Z to process
#if USE_X_PADDING
                my_extended_bounds.padded,       // V13: MY PADDED grid bounds
#else
                my_extended_bounds.core,        // Core grid bounds (no padding)
#endif
                my_pencils,                      // Total pencils in region
                recv_buffer,                     // Source: persistent recv_buffer
                recv_displs_src,                 // Base offsets per source
                src_total_slices,                // Totals per source (for API)
                y_owner_src,                     // Y --> src mapping
                y_src_local_idx,                 // Y --> local_idx mapping
                y_batch_idx,                     // Y --> batch mapping
                y_slice_idx_in_batch,            // Y --> slice_idx in batch mapping
                src_batch_slice_counts,          // [src][batch] --> slice count
                global_max_batches,              // Total number of batches
                local_z_slab,                    // Destination buffer
                thread_1d_bufs,
                num_thread_bufs,
                &acc_unpack,
                &acc_fft,
                plan_1d_y                        // FFT plan (1 FFTW thread; staged)
            );
            
            // ========== DEBUG: Check imaginary parts BEFORE final verification ==========
            #if DEBUG_PRINTS && !SKIP_VERIFICATION
            if (rank == 0 && z == my_extended_bounds.core.z_start) {
                real_t debug_max_real[4] = {0, 0, 0, 0};
                real_t debug_max_imag[4] = {0, 0, 0, 0};
                for (int array_idx = 0; array_idx < narray; array_idx++) {
                    for (int x_idx = 0; x_idx < x_count; x_idx++) {
                        for (int y = 0; y < N; y++) {
                            double re = fabs_t(ZSLAB(array_idx, x_idx, y, N, narray, x_count)[0]);
                            double im = fabs_t(ZSLAB(array_idx, x_idx, y, N, narray, x_count)[1]);
                            debug_max_real[array_idx] = fmax_t(debug_max_real[array_idx], re);
                            debug_max_imag[array_idx] = fmax_t(debug_max_imag[array_idx], im);
                        }
                    }
                }
                printf("[DEBUG Z=%d] After 1D FFT - Max imag per array: ", z);
                for (int a = 0; a < narray; a++) {
                    printf("A%d=%.3e ", a, debug_max_imag[a]);
                }
                printf("\n");
            }
            #endif
            
            // =======================================================================================
            // OUTPUT WRITING (i,j,k notation)
            // =======================================================================================
            // Three output modes:
            //   1. Particle ICs via Option A (transpose + WriteParticlesSlab_range)
            //   2. Particle ICs via Option B (direct WriteParticlesSlab_range_from_zslab)
            //   3. Complex .bin files (fallback when no param_file)
            //      Write this Z-slab in Zeldovich order: [Array][j][k] where:
            //      i = z (Z coordinate, Zeldovich i)
            //      j = y (Y coordinate, Zeldovich j)  
            //      k = x (X coordinate, Zeldovich k)
            //      Each file contains (k_rng, all j) for one i-slab (matches Zeldovich output)
            //      Data is in [Array][k_rng][j] format (memory), transpose to [Array][j][k_rng] for output
            // =======================================================================================

            // Use i,j,k notation for output writing
            int i = z;
            
            // Debug: Check output mode
            if (rank == 0 && z == 0) {
                printf("[OUTPUT-DEBUG] z=%d: param_file=%p, params=%p, PARTICLE_OUTPUT_MODE=%d\n", 
                       z, (void*)param_file, (void*)params, PARTICLE_OUTPUT_MODE);
                fflush(stdout);
            }
            
            // Unified output mode selection (skip when SKIP_FILE_WRITE defined for OMP scaling tests)
#if defined(SKIP_FILE_WRITE) && SKIP_FILE_WRITE
            // Skip all output writes - Stage 3 still does unpack+FFT, but no I/O
            (void)0;
#else
            {
            double t_io0 = omp_get_wtime();
            switch (PARTICLE_OUTPUT_MODE) {
                case 0: {
                    // =======================================================================================
                    // MODE 0: Write particle ICs directly (no transpose needed)
                    // =======================================================================================
                    // Uses WriteParticlesSlab_range with [array][x][y] layout (ZSLAB format)
                    if (param_file == NULL || params == NULL) {
                        break;
                    }
                    
                    int k_start_global = my_extended_bounds.core.x_start;
                    int k_extent = x_count;
                    
                    // Call WriteParticlesSlab_range directly with [array][x][y] layout (ZSLAB format)
                    // No transpose - function handles [x][y] indexing internally
                    WriteParticlesSlab_range(
                        rank, i, k_start_global, k_extent,
                        (Complx*)local_z_slab, N, narray,
                        *static_cast<ZeldovichParameters*>(params)
                    );
                    
                    files_written++;
                    break;
                }
                case 1: {
                    // =======================================================================================
                    // MODE 1: Write .bin files for later re-assembly
                    // =======================================================================================
                    char filename[256];
                    snprintf(filename, sizeof(filename), "rank_%d/i%d_slab_N%d.bin", rank, i, N);
                    
                    FILE *fp = fopen(filename, "wb");
                    if (fp) {
                        // Write in [Array][X][Y] order (no transpose)
                        // Loop order: for (array) for (X) for (Y) to match ZSLAB layout
                        for (int array_idx = 0; array_idx < narray; array_idx++) {
                            for (int k_idx = 0; k_idx < x_count; k_idx++) {
                                for (int j = 0; j < N; j++) {
                                    fftw_complex_t *src = &ZSLAB(array_idx, k_idx, j, N, narray, x_count);
                                    size_t written = fwrite(src, sizeof(fftw_complex_t), 1, fp);
                                    if (written != 1) {
                                        fprintf(stderr, "Rank %d: Write error in %s at (array=%d,k_idx=%d,j=%d)\n",
                                               rank, filename, array_idx, k_idx, j);
                                    }
                                }
                            }
                        }
                        fclose(fp);
                        
                        files_written++;
                        size_t slab_bytes = (size_t)x_count * narray * N * sizeof(fftw_complex_t);
                        total_bytes_written += slab_bytes;
                    } else {
                        fprintf(stderr, "Rank %d: ERROR opening %s for writing (errno=%d)\n", 
                               rank, filename, errno);
                    }
                    break;
                }
                
                case 2: {
                    // =======================================================================================
                    // MODE 2: Write .bin files then immediately read back and write particle ICs
                    // =======================================================================================
                    // Useful for verifying .bin file format and re-assembly logic
                    if (param_file == NULL || params == NULL) {
                        break;
                    }
                    
                    // Step 1: Write .bin file (same as Mode 2)
                    char filename[256];
                    snprintf(filename, sizeof(filename), "rank_%d/i%d_slab_N%d.bin", rank, i, N);
                    
                    FILE *fp = fopen(filename, "wb");
                    if (!fp) {
                        fprintf(stderr, "Rank %d: ERROR opening %s for writing (errno=%d)\n", 
                               rank, filename, errno);
                        break;
                    }
                    
                    // Transpose from [Array][X][Y] to [Array][Y][X] and write to .bin file
                    for (int array_idx = 0; array_idx < narray; array_idx++) {
                        for (int j = 0; j < N; j++) {
                            for (int k_idx = 0; k_idx < x_count; k_idx++) {
                                fftw_complex_t *src = &ZSLAB(array_idx, k_idx, j, N, narray, x_count);
                                size_t written = fwrite(src, sizeof(fftw_complex_t), 1, fp);
                                if (written != 1) {
                                    fprintf(stderr, "Rank %d: Write error in %s at (array=%d,j=%d,k_idx=%d)\n",
                                           rank, filename, array_idx, j, k_idx);
                                }
                            }
                        }
                    }
                    fclose(fp);
                    
                    // Step 2: Read .bin file back into [y][x] format and call WriteParticlesSlab_range
                    fp = fopen(filename, "rb");
                    if (!fp) {
                        fprintf(stderr, "Rank %d: ERROR opening %s for reading (errno=%d)\n", 
                               rank, filename, errno);
                        break;
                    }
                    fftw_complex_t *T_slab1 = (fftw_complex_t*)FFTW_MALLOC(x_count * N * sizeof(fftw_complex_t));
                    fftw_complex_t *T_slab2 = (fftw_complex_t*)FFTW_MALLOC(x_count * N * sizeof(fftw_complex_t));
                    fftw_complex_t *T_slab3 = (fftw_complex_t*)FFTW_MALLOC(x_count * N * sizeof(fftw_complex_t));
                    fftw_complex_t *T_slab4 = (fftw_complex_t*)FFTW_MALLOC(x_count * N * sizeof(fftw_complex_t));
                    
                    if (!T_slab1 || !T_slab2 || !T_slab3 || !T_slab4) {
                        fprintf(stderr, "Rank %d: ERROR: Failed to allocate read buffers for i=%d\n", rank, i);
                        fclose(fp);
                        break;
                    }
                    
                    // Read in [Array][Y][X] order (as written)
                    for (int array_idx = 0; array_idx < narray; array_idx++) {
                        fftw_complex_t *dest = (array_idx == 0) ? T_slab1 :
                                               (array_idx == 1) ? T_slab2 :
                                               (array_idx == 2) ? T_slab3 : T_slab4;
                        
                        for (int j = 0; j < N; j++) {
                            for (int k_idx = 0; k_idx < x_count; k_idx++) {
                                size_t read = fread(&dest[j * x_count + k_idx], sizeof(fftw_complex_t), 1, fp);
                                if (read != 1) {
                                    fprintf(stderr, "Rank %d: Read error in %s at (array=%d,j=%d,k_idx=%d)\n",
                                           rank, filename, array_idx, j, k_idx);
                                }
                            }
                        }
                    }
                    fclose(fp);
                    
                    // Step 3: Call WriteParticlesSlab_range
                    int k_start_global = my_extended_bounds.core.x_start;
                    int k_extent = x_count;
                    
                    WriteParticlesSlab_range(
                        rank, i, k_start_global, k_extent,
                        (Complx*)T_slab1, (Complx*)T_slab2, (Complx*)T_slab3, (Complx*)T_slab4,
                        *static_cast<ZeldovichParameters*>(params)
                    );
                    
                    FFTW_FREE(T_slab1);
                    FFTW_FREE(T_slab2);
                    FFTW_FREE(T_slab3);
                    FFTW_FREE(T_slab4);
                    
                    files_written++;
                    break;
                }
                
                case 3: {
                    if (grid_x == 1) {
                        // ===========================================================================
                        // grid_x==1: z-group format — one full NxN plane per z (zeldovich-compatible)
                        // ===========================================================================
                        int file_index = z * cpd / N;
                        FILE *fp = zgrp_fp[file_index - zgrp_start];
                        FILE *fp_dens = (file_index - zgrp_start < (int)zgrp_dens_fp.size())
                                        ? zgrp_dens_fp[file_index - zgrp_start] : NULL;
                        if (fp != NULL) {
                            AppendZSlabFull(
                                fp, fp_dens, z,
                                my_extended_bounds.core.x_start, x_count,
                                local_z_slab, N, narray,
                                *static_cast<ZeldovichParameters*>(params)
                            );
                        }
                        total_bytes_written += (size_t)N * N * sizeof(RVZelParticle);
                        if (static_cast<ZeldovichParameters*>(params)->qdensity)
                            total_bytes_written += (size_t)N * N * sizeof(float);
                    } else {
                        // ===========================================================================
                        // grid_x>1: ic_s is Abacus x-slab s; per z append to every ic_s in rank band
                        // ===========================================================================
                        for (int s = slab_x_start; s < slab_x_end; s++) {
                            FILE *fp = slab_fp[s - slab_x_start];
                            FILE *fp_dens = (s - slab_x_start < (int)slab_dens_fp.size())
                                            ? slab_dens_fp[s - slab_x_start] : NULL;
                            if (fp != NULL) {
                                if (z == 0) {
                                    fprintf(stderr,
                                            "[IC_WRITE_DEBUG] MPI_rank=%d z=%d append ic_%04d\n",
                                            rank, z, s);
                                    fflush(stderr);
                                }
                                AppendSlabZSegment(
                                    fp, fp_dens, s, cpd, z,
                                    my_extended_bounds.core.x_start, x_count,
                                    local_z_slab, N, narray,
                                    *static_cast<ZeldovichParameters*>(params)
                                );
                            }
                            int firstx = (s * N + cpd - 1) / cpd;
                            int lastx  = ((s + 1) * N + cpd - 1) / cpd;
                            int k_start = my_extended_bounds.core.x_start;
                            int seg_start = std::max(firstx, k_start);
                            int seg_end   = std::min(lastx, k_start + x_count);
                            if (seg_start < seg_end) {
                                int ox_count = seg_end - seg_start;
                                total_bytes_written += (size_t)ox_count * N * sizeof(RVZelParticle);
                                if (static_cast<ZeldovichParameters*>(params)->qdensity)
                                    total_bytes_written += (size_t)ox_count * N * sizeof(float);
                            }
                        }
                    }
                    break;
                }

                case 4: {
                    // ===========================================================================
                    // MODE 4: z-slab format — one segment per z to the owning z-slab file
                    // ===========================================================================
                    int s = z * cpd / N;
                    if (s >= slab_z_start && s < slab_z_end) {
                        FILE *fp = zslab_fp[s - slab_z_start];
                        FILE *fp_dens = (s - slab_z_start < (int)zslab_dens_fp.size())
                                        ? zslab_dens_fp[s - slab_z_start] : NULL;
                        if (fp != NULL) {
                            AppendZSlabSegment_M4(
                                fp, fp_dens, z,
                                my_extended_bounds.core.x_start, x_count,
                                local_z_slab, N, narray,
                                *static_cast<ZeldovichParameters*>(params)
                            );
                        }
                    }
                    int ox_total = my_extended_bounds.core.x_end - my_extended_bounds.core.x_start;
                    total_bytes_written += (size_t)ox_total * N * sizeof(RVZelParticle);
                    if (static_cast<ZeldovichParameters*>(params)->qdensity)
                        total_bytes_written += (size_t)ox_total * N * sizeof(float);
                    break;
                }
                
                default:
                    break;
            }
            acc_io += omp_get_wtime() - t_io0;
            }
#endif // !SKIP_FILE_WRITE

            // =======================================================================================
            
            // Progress indicator for large grids
            if (DEBUG_PRINTS && rank == 0 && z_count > 10) {
#if USE_X_PADDING
                int z_processed = z - my_extended_bounds.padded.z_start + 1;
#else
                int z_processed = z - my_extended_bounds.core.z_start + 1;
#endif
                if (z_processed % (z_count / 10) == 0 || z_processed == z_count) {
                    printf("         ... Processed %d/%d Z-slabs (%.1f%%)\n", 
                           z_processed, z_count, 100.0 * z_processed / z_count);
                }
            }
        }

        for (int t = 0; t < num_threads; t++) {
            if (thread_1d_bufs[t] != NULL) {
                free(thread_1d_bufs[t]);
                thread_1d_bufs[t] = NULL;
            }
        }
        free(thread_1d_bufs);
    }
    // Idle ranks: files_written = 0, total_bytes_written = 0 (already initialized)
    
    // MODE 3: Close files and set file count
    if (PARTICLE_OUTPUT_MODE == 3 && params != NULL) {
        if (grid_x == 1) {
            for (size_t i = 0; i < zgrp_fp.size(); i++) {
                if (zgrp_fp[i] != NULL) { fclose(zgrp_fp[i]); zgrp_fp[i] = NULL; }
                if (i < zgrp_dens_fp.size() && zgrp_dens_fp[i] != NULL) {
                    fclose(zgrp_dens_fp[i]); zgrp_dens_fp[i] = NULL;
                }
            }
            files_written = zgrp_end - zgrp_start;
        } else {
            for (size_t i = 0; i < slab_fp.size(); i++) {
                if (slab_fp[i] != NULL) { fclose(slab_fp[i]); slab_fp[i] = NULL; }
                if (i < slab_dens_fp.size() && slab_dens_fp[i] != NULL) {
                    fclose(slab_dens_fp[i]); slab_dens_fp[i] = NULL;
                }
            }
            files_written = slab_x_end - slab_x_start;
        }
        MPI_Barrier(zd_comm_2d);
    }

    // MODE 4: Close files and set file count
    if (PARTICLE_OUTPUT_MODE == 4 && params != NULL) {
        for (size_t i = 0; i < zslab_fp.size(); i++) {
            if (zslab_fp[i] != NULL) { fclose(zslab_fp[i]); zslab_fp[i] = NULL; }
            if (i < zslab_dens_fp.size() && zslab_dens_fp[i] != NULL) {
                fclose(zslab_dens_fp[i]); zslab_dens_fp[i] = NULL;
            }
        }
        files_written = slab_z_end - slab_z_start;
        MPI_Barrier(zd_comm_2d);
    }
    
    t_streaming.Stop();

    double acc_unpack_max = 0.0, acc_fft_max = 0.0, acc_io_max = 0.0;
    MPI_Reduce(&acc_unpack, &acc_unpack_max, 1, MPI_DOUBLE, MPI_MAX, 0, zd_comm_2d);
    MPI_Reduce(&acc_fft, &acc_fft_max, 1, MPI_DOUBLE, MPI_MAX, 0, zd_comm_2d);
    MPI_Reduce(&acc_io, &acc_io_max, 1, MPI_DOUBLE, MPI_MAX, 0, zd_comm_2d);
    
    int total_files_written;
    size_t total_bytes_all_ranks;
    MPI_Reduce(&files_written, &total_files_written, 1, MPI_INT, MPI_SUM, 0, zd_comm_2d);
    MPI_Reduce(&total_bytes_written, &total_bytes_all_ranks, 1, MPI_UNSIGNED_LONG, 
               MPI_SUM, 0, zd_comm_2d);

    if (rank == 0) {
        printf("[Stage 3] Streaming complete. Time: %.6f s\n", t_streaming.Elapsed());
        printf("          (Includes unpacking, FFT, and I/O for all Z-slabs)\n");
        printf("          Unpack (recv -> slab, max rank): %.6f s\n", acc_unpack_max);
        printf("          1D Y FFT (staged, max rank):    %.6f s\n", acc_fft_max);
        printf("          Write / I/O (switch, max rank):  %.6f s\n", acc_io_max);
        printf("          Total files written: %d (across all ranks)\n", total_files_written);
        printf("          Total data written: %.3f GB\n", 
               total_bytes_all_ranks / (1024.0 * 1024.0 * 1024.0));
        
        // Calculate typical file size (varies by rank due to grid decomposition)
        // Each rank writes z_count files (one per Z-slab it owns)
        // Each file contains: x_count X x narray arrays x N Y-values
        int typical_x_count = N / (int)sqrt((double)num_ranks);
        size_t typical_file_size = (size_t)typical_x_count * narray * N * sizeof(fftw_complex_t);
        int typical_files_per_rank = total_files_written / num_ranks;
        printf("          Files per rank: ~%d (z_count per rank, for %d ranks)\n", 
               typical_files_per_rank, num_ranks);
        printf("          Each file contains one Z-slab: x_count X x %d arrays x %d Y-values\n", 
               narray, N);
        printf("          Typical file size: ~%.2f MB (varies by rank's X-extent)\n",
               typical_file_size / (1024.0 * 1024.0));
        printf("          File naming: rank_<rank>/z<z>_slab_N%d.bin\n", N);
        printf("          File format: Binary [Array][Y][X] (Zeldovich AZYX order, fftw_complex = %d bytes)\n", BYTES_PER_COMPLEX);
        printf("          3D FFT COMPLETE (2D XZ + 1D Y)\n\n");
    }
    
    // NOTE: In multi-batch mode, communication buffers are freed per-batch within the batch loop
    // (sendcounts_batch, sdispls_batch, recvcounts_batch)
    
    // ========================================================================
    // TIMING SUMMARY
    // ========================================================================
    
    #if DETAILED_TIMING
    if (rank == 0) {
        printf("\n====================================================================================\n");
        printf("TIMING SUMMARY (v11 - Persistent Recv Buffer + Streaming)\n");
        printf("====================================================================================\n");
        printf("Stage 1 (Y-slice generation + 2D FFT): %.6f s\n", t_gen.Elapsed());
        printf("Stage 2 (Metadata exchange):            %.6f s\n", 0.0);  // Minimal time
        printf("Stage 3 (Communication: Alltoallv):    %.6f s\n", t_comm.Elapsed());
        printf("Stage 4 (Streaming: Unpack+FFT+Write):  %.6f s\n", t_streaming.Elapsed());
        printf("  (max rank) Unpack / 1D-Y-FFT / I/O:   %.6f / %.6f / %.6f s\n",
               acc_unpack_max, acc_fft_max, acc_io_max);
        printf("------------------------------------------------------------------------------------\n");
        printf("Total 3D FFT time (Gen + Comm + FFT):   %.6f s\n", 
               t_gen.Elapsed() + t_comm.Elapsed() + t_streaming.Elapsed());
        printf("Total time (including all stages):      %.6f s\n", 
               t_gen.Elapsed() + t_comm.Elapsed() + t_streaming.Elapsed());
        printf("====================================================================================\n");
    }
    #endif
    
    FFTW_DESTROY_PLAN(plan_2d);
    FFTW_DESTROY_PLAN(plan_1d_y);
    
    // FREE: local_y_slices (normally freed after each batch pack; safety check)
    if (!is_idle_rank && local_y_slices != NULL) {
        fprintf(stderr, "[WARNING] Rank %d: local_y_slices still allocated at final cleanup\n", rank);
        free(local_y_slices);
        local_y_slices = NULL;
    }
    if (!is_idle_rank && fft_stage_2d != NULL) {
        fprintf(stderr, "[WARNING] Rank %d: fft_stage_2d was not freed earlier!\n", rank);
        free(fft_stage_2d);
        fft_stage_2d = NULL;
    }
    if (!is_idle_rank && local_z_slab != NULL) {
        free(local_z_slab);
        local_z_slab = NULL;
    }
    // FREE: Y-mapping arrays (used for unpacking)
    if (y_owner_src != NULL) {
        free(y_owner_src);
        y_owner_src = NULL;
    }
    if (y_src_local_idx != NULL) {
        free(y_src_local_idx);
        y_src_local_idx = NULL;
    }
    if (y_batch_idx != NULL) {
        free(y_batch_idx);
        y_batch_idx = NULL;
    }
    if (y_slice_idx_in_batch != NULL) {
        free(y_slice_idx_in_batch);
        y_slice_idx_in_batch = NULL;
    }
    if (src_batch_slice_counts != NULL) {
        for (int src = 0; src < num_ranks; src++) {
            if (src_batch_slice_counts[src] != NULL) {
                free(src_batch_slice_counts[src]);
            }
        }
        free(src_batch_slice_counts);
        src_batch_slice_counts = NULL;
    }
    if (recv_buffer != NULL) {
        free(recv_buffer);
        recv_buffer = NULL;
    }
    if (recv_displs_src != NULL) {
        free(recv_displs_src);
        recv_displs_src = NULL;
    }
    if (src_total_slices != NULL) {
        free(src_total_slices);
        src_total_slices = NULL;
    }
    if (src_write_cursor != NULL) {
        free(src_write_cursor);
        src_write_cursor = NULL;
    }
    if (my_pair_list != NULL) {
        free(my_pair_list);
        my_pair_list = NULL;
    }
    cleanup_global_pcg();
    if (ps) {
        zeldovich_ps_destroy(ps);
        ps = NULL;
    }
    if (ps_primordial) {
        zeldovich_ps_destroy(ps_primordial);
        ps_primordial = NULL;
    }
    TeardownOutput();
    if (params) {
        zeldovich_params_destroy(params);
        params = NULL;
    }

    // Free PLT eigenmodes if they were loaded
    plt_free_eigenmodes();

    /* Plans destroyed above; tear down threaded FFTW state for given precision (standalone + embedded). */
#ifdef USE_DOUBLE_PRECISION
    fftw_cleanup_threads();
#else
    fftwf_cleanup_threads();
#endif

    if (zd_comm_2d != MPI_COMM_NULL) {
        MPI_Comm_free(&zd_comm_2d);
        zd_comm_2d = MPI_COMM_NULL;
    }

    if (rank == 0) {
        printf("\n====================================================================================\n");
        printf("ALL STAGES COMPLETE!\n");
        printf("====================================================================================\n");
    }

    return 0;
}