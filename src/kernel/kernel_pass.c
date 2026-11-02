// ====================================================================================
// ONE KERNEL TERM (see kernel_pass.h for the stage overview)
// ====================================================================================

#include "kernel_pass.h"
#include "communication/mpi_exchange.h"
#include "fft/fft_setup.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KERNEL_PAGE 4096
#define KERNEL_ROUND_UP_PAGE(x) ((((size_t)(x)) + KERNEL_PAGE - 1) & ~((size_t)(KERNEL_PAGE - 1)))

fftw_complex_t *kernel_alloc_complex(int64_t count, const char *what)
{
    if (count <= 0) count = 1;
    size_t bytes = KERNEL_ROUND_UP_PAGE((size_t)count * sizeof(fftw_complex_t));
    fftw_complex_t *p = NULL;
    if (posix_memalign((void**)&p, ALIGN_BYTES, bytes) != 0) {
        fprintf(stderr, "[Kernel_MPI] ERROR: posix_memalign failed for %s (%.3f GB)\n", what, bytes / 1.0e9);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return p;
}

// ====================================================================================
// Alltoallv with int64 counts/displacements (zd_alltoallv_int64): MPI_Alltoallv_c on
// MPI >= 4, Isend/Irecv at int64 offsets otherwise. KERNEL_FORCE_STAGED_A2A=1 forces the
// point-to-point path even on MPI >= 4 (used by the regression tests).
// ====================================================================================

static void kernel_alltoallv(const fftw_complex_t *sendbuf, const int64_t *sc, const int64_t *sd,
                             fftw_complex_t *recvbuf, const int64_t *rc, const int64_t *rd,
                             int num_ranks, MPI_Comm comm)
{
    static int force_p2p = -1;
    if (force_p2p < 0) {
        const char *v = getenv("KERNEL_FORCE_STAGED_A2A");
        force_p2p = (v != NULL && atoi(v) != 0);
    }

    int err = zd_alltoallv_int64(sendbuf, sc, sd, recvbuf, rc, rd,
                                 MPI_COMPLEX_TYPE, num_ranks, comm, force_p2p);
    if (err != MPI_SUCCESS) {
        char errstr[MPI_MAX_ERROR_STRING];
        int errlen = 0;
        MPI_Error_string(err, errstr, &errlen);
        fprintf(stderr, "[Kernel_MPI] ERROR: Alltoallv failed: %s\n", errstr);
        MPI_Abort(MPI_COMM_WORLD, err);
    }
}

// ====================================================================================
// Setup: FFT plans (backward from fft_setup.c, forward planned the same way) + buffers
// ====================================================================================

void kernel_pass_init(KernelPass *P, const KernelLayout *K, MPI_Comm comm, double box_size,
                      fftw_complex_t **phi, fftw_complex_t **out)
{
    memset(P, 0, sizeof(*P));
    P->K = K;
    P->comm = comm;
    P->kfund = 2.0 * M_PI / box_size;
    P->phi = phi;
    P->out = out;

    int M = K->M;
    int64_t plane = (int64_t)M * M;

    // 2D plans need an M x M buffer; 1D plan uses a dummy line (staged execution not needed:
    // every plane/line we execute on is ALIGN_BYTES-aligned or a multiple of
    // sizeof(fftw_complex_t)*M bytes in)
    fftw_complex_t *plan_buf = kernel_alloc_complex(plane, "fft plan buffer");
    memset(plan_buf, 0, sizeof(fftw_complex_t) * (size_t)plane);
    setup_fftw_plans_full(M, 1, plan_buf, &P->plan2d_bwd, &P->plan1d_bwd, NULL);

    FFTW_PLAN_WITH_NTHREADS(omp_get_max_threads());
    P->plan2d_fwd = FFTW_PLAN_DFT_2D(M, M, plan_buf, plan_buf, FFTW_FORWARD, FFTW_PLANNER_FLAGS);
    FFTW_PLAN_WITH_NTHREADS(1);
    fftw_complex_t *line = kernel_alloc_complex(M, "fft 1d plan buffer");
    P->plan1d_fwd = FFTW_PLAN_DFT_1D(M, line, line, FFTW_FORWARD, FFTW_PLANNER_FLAGS);
    free(line);
    free(plan_buf);

    if (P->plan2d_fwd == NULL || P->plan1d_fwd == NULL) {
        fprintf(stderr, "[Kernel_MPI] ERROR: failed to create forward FFT plans\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // q_half is written in place over recv_buffer. recv_buffer is [slot][pencil] with N
    // slots of my_pencils each (every offset in kernel_layout_init is a multiple of
    // my_pencils), and q_half is [j][pencil] with j < N/2+1 <= N slots. In stage B a pencil
    // reads all N of its slots into a line before writing its N/2+1 values back, and
    // different pencils touch disjoint elements, so no unread data is overwritten.
    // Stage C only reads q_half, and the next term's stage A refills recv_buffer after it.
    if (K->recv_total != (int64_t)K->N * K->my_pencils) {
        fprintf(stderr, "[Kernel_MPI] ERROR: recv_total %lld != N * my_pencils %lld\n",
                (long long)K->recv_total, (long long)K->N * K->my_pencils);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    P->recv_buffer = kernel_alloc_complex(K->recv_total, "forward recv_buffer");
    P->q_half = P->recv_buffer;
}

void kernel_pass_free(KernelPass *P)
{
    FFTW_DESTROY_PLAN(P->plan2d_bwd);
    FFTW_DESTROY_PLAN(P->plan2d_fwd);
    FFTW_DESTROY_PLAN(P->plan1d_bwd);
    FFTW_DESTROY_PLAN(P->plan1d_fwd);
    free(P->recv_buffer);   // q_half aliases it
}

// ====================================================================================
// Stage A: y-slices (k space) -> (G + iH)*phi padded -> 2D bwd FFT -> Alltoallv
// ====================================================================================

static void stage_forward(KernelPass *P, const KernelShape *shape)
{
    const KernelLayout *K = P->K;
    const int N = K->N, M = K->M, R = K->num_ranks, rank = K->rank;
    const int64_t plane = (int64_t)M * M;

    int64_t *sendcounts = (int64_t*)calloc(R, sizeof(int64_t));
    int64_t *sdispls    = (int64_t*)calloc(R, sizeof(int64_t));
    int64_t *recvcounts = (int64_t*)calloc(R, sizeof(int64_t));
    int64_t *rdispls    = (int64_t*)calloc(R, sizeof(int64_t));
    int64_t *cursor     = (int64_t*)calloc(R, sizeof(int64_t));

    // [slice][z][x] on the padded grid (Y_SLICE layout with narray = 1)
    fftw_complex_t *local_y_slices = kernel_alloc_complex(2 * plane, "local_y_slices");
    fftw_complex_t *send_buffer = kernel_alloc_complex(2 * plane, "forward send_buffer");

    for (int b = 0; b < K->num_batches; b++) {
        int count = K->src_batch_slice_counts[rank][b];
        int y_p = kernel_batch_y_p(K, rank, b);

        double t0 = MPI_Wtime();
        if (count > 0) {
            memset(local_y_slices, 0, sizeof(fftw_complex_t) * (size_t)(count * plane));
            const fftw_complex_t *src = P->phi[b];

            for (int s = 0; s < count; s++) {
                int y_orig = (s == 0) ? y_p : N - y_p;   // primary, then its mirror
                double ky = (double)kfreq(y_orig, N);
                fftw_complex_t *dst = local_y_slices + s * plane;

                // Pad: loop over the original N x N modes, write at padded positions
                #pragma omp parallel for schedule(static)
                for (int iz = 0; iz < N; iz++) {
                    double kz = (double)kfreq(iz, N);
                    int pz = pad_idx(iz, N, M);
                    int iz_m = (N - iz) % N;
                    for (int ix = 0; ix < N; ix++) {
                        double vr, vi;
                        if (s == 0) {
                            vr = src[(size_t)iz * N + ix][0];
                            vi = src[(size_t)iz * N + ix][1];
                        } else {
                            // phi(x, N - y_p, z) = conj(phi(-x, y_p, -z))
                            int ix_m = (N - ix) % N;
                            vr =  src[(size_t)iz_m * N + ix_m][0];
                            vi = -src[(size_t)iz_m * N + ix_m][1];
                        }
                        double kx = (double)kfreq(ix, N);
                        double k = P->kfund * sqrt(kx * kx + ky * ky + kz * kz);
                        double g = kpow(k, shape->pG);
                        double h = kpow(k, shape->pH);
                        // G*phi + i*H*phi
                        fftw_complex_t *d = &dst[(size_t)pz * M + pad_idx(ix, N, M)];
                        (*d)[0] = (real_t)(g * vr - h * vi);
                        (*d)[1] = (real_t)(g * vi + h * vr);
                    }
                }
                FFTW_EXECUTE_DFT(P->plan2d_bwd, dst, dst);
            }
        }
        double t1 = MPI_Wtime();

        if (count > 0) {
            // Fills sendcounts/sdispls: [array][slice][x][z] per destination
            pack_slices_to_send_buffer(rank, R, M, 1, local_y_slices, count, NULL,
                                       send_buffer, sendcounts, sdispls,
                                       K->size_x, K->size_z, M);
        } else {
            for (int d = 0; d < R; d++) { sendcounts[d] = 0; sdispls[d] = 0; }
        }
        for (int src = 0; src < R; src++) {
            recvcounts[src] = (int64_t)K->src_batch_slice_counts[src][b] * K->my_pencils;
            rdispls[src] = K->recv_displs_src[src] + cursor[src];
        }
        double t2 = MPI_Wtime();

        kernel_alltoallv(send_buffer, sendcounts, sdispls,
                         P->recv_buffer, recvcounts, rdispls, R, P->comm);
        for (int src = 0; src < R; src++) cursor[src] += recvcounts[src];
        double t3 = MPI_Wtime();

        P->t_fill_fft2d += t1 - t0;
        P->t_pack += t2 - t1;
        P->t_comm_fwd += t3 - t2;
    }

    for (int src = 0; src < R; src++) {
        int64_t expected = 0;
        for (int b = 0; b < K->num_batches; b++) {
            expected += (int64_t)K->src_batch_slice_counts[src][b] * K->my_pencils;
        }
        if (cursor[src] != expected) {
            fprintf(stderr, "[Kernel_MPI] ERROR: rank %d src %d cursor %lld != expected %lld\n",
                    rank, src, (long long)cursor[src], (long long)expected);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    free(local_y_slices);
    free(send_buffer);
    free(sendcounts); free(sdispls); free(recvcounts); free(rdispls); free(cursor);
}

// ====================================================================================
// Stage B: per z-slab, unpack -> 1D bwd FFT (y) -> Q = g*h -> 1D fwd FFT -> y crop
// ====================================================================================

static void stage_zslab(KernelPass *P)
{
    const KernelLayout *K = P->K;
    const int N = K->N, M = K->M;
    const int x_count = K->x_count, z_count = K->z_count;
    const int64_t my_pencils = K->my_pencils;
    const double inv_M3 = 1.0 / ((double)M * (double)M * (double)M);

    double t0 = MPI_Wtime();
    // One y line per x in my block: [x][y] (y stride-1 for the 1D FFT)
    fftw_complex_t *local_z_slab = kernel_alloc_complex((int64_t)x_count * M, "local_z_slab");

    for (int z_idx = 0; z_idx < z_count; z_idx++) {
        #pragma omp parallel for schedule(static)
        for (int x_idx = 0; x_idx < x_count; x_idx++) {
            fftw_complex_t *line = local_z_slab + (int64_t)x_idx * M;
            int64_t pencil_idx = (int64_t)x_idx * z_count + z_idx;

            // Unpack (padded y rows are zero)
            for (int y = 0; y < M; y++) {
                int64_t base = K->y_base[y];
                if (base < 0) {
                    line[y][0] = 0.0;
                    line[y][1] = 0.0;
                } else {
                    line[y][0] = P->recv_buffer[base + pencil_idx][0];
                    line[y][1] = P->recv_buffer[base + pencil_idx][1];
                }
            }

            FFTW_EXECUTE_DFT(P->plan1d_bwd, line, line);

            // z = g + i h with g, h real  =>  g*h = Im(z^2)/2 ; single 1/M^3 normalization
            for (int y = 0; y < M; y++) {
                double q = (double)line[y][0] * (double)line[y][1] * inv_M3;
                line[y][0] = (real_t)q;
                line[y][1] = 0.0;
            }

            FFTW_EXECUTE_DFT(P->plan1d_fwd, line, line);

            // Crop in y and keep the Hermitian half: y_p in [0, N/2] only
            for (int j = 0; j <= N / 2; j++) {
                int yp = pad_idx(j, N, M);
                P->q_half[(int64_t)j * my_pencils + pencil_idx][0] = line[yp][0];
                P->q_half[(int64_t)j * my_pencils + pencil_idx][1] = line[yp][1];
            }
        }
    }

    free(local_z_slab);
    P->t_zslab += MPI_Wtime() - t0;
}

// ====================================================================================
// Stage C: q_half column -> y_p owner -> M x M plane -> 2D fwd FFT -> crop * J -> out
// ====================================================================================

static void stage_return(KernelPass *P, const KernelShape *shape, double coef)
{
    const KernelLayout *K = P->K;
    const int N = K->N, M = K->M, R = K->num_ranks, rank = K->rank;
    const int64_t plane = (int64_t)M * M;
    const int64_t my_pencils = K->my_pencils;

    int64_t *sendcounts = (int64_t*)calloc(R, sizeof(int64_t));
    int64_t *sdispls    = (int64_t*)calloc(R, sizeof(int64_t));
    int64_t *recvcounts = (int64_t*)calloc(R, sizeof(int64_t));
    int64_t *rdispls    = (int64_t*)calloc(R, sizeof(int64_t));

    fftw_complex_t *send_buffer = kernel_alloc_complex((int64_t)R * my_pencils, "return send_buffer");
    fftw_complex_t *recv_buffer = kernel_alloc_complex(plane, "return recv_buffer");
    fftw_complex_t *q_plane = kernel_alloc_complex(plane, "return q_plane");

    for (int b = 0; b < K->num_batches; b++) {
        int my_y_p = kernel_batch_y_p(K, rank, b);

        double t0 = MPI_Wtime();
        int64_t so = 0;
        for (int d = 0; d < R; d++) {
            int y_p = kernel_batch_y_p(K, d, b);
            sdispls[d] = so;
            if (y_p >= 0) {
                memcpy(send_buffer + so, P->q_half + (int64_t)y_p * my_pencils,
                       sizeof(fftw_complex_t) * (size_t)my_pencils);
                sendcounts[d] = my_pencils;
                so += my_pencils;
            } else {
                sendcounts[d] = 0;
            }
        }
        int64_t ro = 0;
        for (int src = 0; src < R; src++) {
            recvcounts[src] = (my_y_p >= 0) ? kernel_region_size(&K->bounds[src]) : 0;
            rdispls[src] = ro;
            ro += recvcounts[src];
        }

        kernel_alltoallv(send_buffer, sendcounts, sdispls, recv_buffer, recvcounts, rdispls, R, P->comm);
        double t1 = MPI_Wtime();
        P->t_comm_ret += t1 - t0;

        if (my_y_p < 0) continue;

        // Scatter each source's pencils (pencil = x_idx * z_count + z_idx) into the [z][x] plane
        for (int src = 0; src < R; src++) {
            const GridBounds *g = &K->bounds[src];
            int xc = g->x_end - g->x_start;
            int zc = g->z_end - g->z_start;
            const fftw_complex_t *blk = recv_buffer + rdispls[src];
            #pragma omp parallel for schedule(static)
            for (int z_idx = 0; z_idx < zc; z_idx++) {
                fftw_complex_t *row = q_plane + (int64_t)(g->z_start + z_idx) * M + g->x_start;
                for (int x_idx = 0; x_idx < xc; x_idx++) {
                    row[x_idx][0] = blk[(int64_t)x_idx * zc + z_idx][0];
                    row[x_idx][1] = blk[(int64_t)x_idx * zc + z_idx][1];
                }
            }
        }

        FFTW_EXECUTE_DFT(P->plan2d_fwd, q_plane, q_plane);

        // Crop: read only the original N x N modes; out += coef * (Q * J)
        fftw_complex_t *out = P->out[b];
        double ky = (double)kfreq(my_y_p, N);
        #pragma omp parallel for schedule(static)
        for (int iz = 0; iz < N; iz++) {
            double kz = (double)kfreq(iz, N);
            const fftw_complex_t *row = q_plane + (int64_t)pad_idx(iz, N, M) * M;
            for (int ix = 0; ix < N; ix++) {
                if (my_y_p == 0 && iz == 0 && ix == 0) continue;   // DC mode of Q set to zero
                double kx = (double)kfreq(ix, N);
                double k = P->kfund * sqrt(kx * kx + ky * ky + kz * kz);
                double J = shape->cJ * kpow(k, shape->pJ);
                const fftw_complex_t *q = &row[pad_idx(ix, N, M)];
                out[(size_t)iz * N + ix][0] += (real_t)(coef * ((*q)[0] * J));
                out[(size_t)iz * N + ix][1] += (real_t)(coef * ((*q)[1] * J));
            }
        }
        P->t_fft2d_accum += MPI_Wtime() - t1;
    }

    free(send_buffer);
    free(recv_buffer);
    free(q_plane);
    free(sendcounts); free(sdispls); free(recvcounts); free(rdispls);
}

void run_kernel_term(KernelPass *P, const KernelShape *shape, double coef)
{
    stage_forward(P, shape);
    stage_zslab(P);
    stage_return(P, shape, coef);
}
