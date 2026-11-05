// ====================================================================================
// KERNEL TERM TABLES

// ====================================================================================

#include "kernel_terms.h"
#include <string.h>

// Each shape is {name, pG, pH, pJ, cJ}: G = |k|^pG, H = |k|^pH, J = cJ * |k|^pJ,
// contributing J(k) * FFT[ IFFT(G*phi) * IFFT(H*phi) ].
// E.g. C1 = {1, 1, -2, 1/3}: G = k, H = k, J = 1/(3k^2).
static const KernelShape kernel_shapes[KSHAPE_COUNT] = {
    [KSHAPE_A1] = {"A1",  0,  0,  0, 1.0},
    [KSHAPE_C1] = {"C1",  1,  1, -2, 1.0 / 3.0},
    [KSHAPE_B1] = {"B1",  1,  0, -1, 1.0},
    [KSHAPE_B2] = {"B2",  2,  0, -2, 1.0},
    [KSHAPE_D1] = {"D1", -1, -1,  2, 1.0},
    [KSHAPE_E2] = {"E2",  2, -1, -1, 1.0 / 2.0},
    [KSHAPE_F1] = {"F1", -1,  0,  1, 1.0},
    [KSHAPE_G1] = {"G1", -1,  1,  0, 1.0},
};

const KernelShape *kernel_shape(KernelShapeId id)
{
    return &kernel_shapes[id];
}

int kernel_shape_from_name(const char *name)
{
    for (int i = 0; i < KSHAPE_COUNT; i++) {
        if (strcmp(name, kernel_shapes[i].name) == 0) return i;
    }
    return -1;
}

const char *fnl_type_name(FnlType type)
{
    switch (type) {
        case FNL_LOCAL:    return "local";
        case FNL_EQUIL:    return "equil";
        case FNL_ORTHOCMB: return "orthoCMB";
        case FNL_ORTHOLSS: return "orthoLSS";
        default:           return "none";
    }
}

int kernel_terms_for(FnlType type, KernelTerm *terms)
{
    int n = 0;
    switch (type) {
        case FNL_LOCAL:
            terms[n++] = (KernelTerm){KSHAPE_A1, 1.0};
            break;

        // kernel_equil = -3*K_A1 - 6*K_C1 + 4*K_B1 + 2*K_B2
        case FNL_EQUIL:
            terms[n++] = (KernelTerm){KSHAPE_A1, -3.0};
            terms[n++] = (KernelTerm){KSHAPE_C1, -6.0};
            terms[n++] = (KernelTerm){KSHAPE_B1,  4.0};
            terms[n++] = (KernelTerm){KSHAPE_B2,  2.0};
            break;

        // kernel_orthoCMB = -9*K_A1 - 24*K_C1 + 10*K_B1 + 8*K_B2
        case FNL_ORTHOCMB:
            terms[n++] = (KernelTerm){KSHAPE_A1,  -9.0};
            terms[n++] = (KernelTerm){KSHAPE_C1, -24.0};
            terms[n++] = (KernelTerm){KSHAPE_B1,  10.0};
            terms[n++] = (KernelTerm){KSHAPE_B2,   8.0};
            break;

        // Derived assuming a scale invariant spectrum (n2=1), as in kernel_MPI.py
        case FNL_ORTHOLSS: {
            double p = 27.0 / (-21.0 + (743.0) / (7.0 * (20.0 * M_PI * M_PI - 193.0)));
            double t = (2.0 + (60.0 * p) / 27.0) / (6.0 * (1.0 + (15.0 * p) / 27.0));
            terms[n++] = (KernelTerm){KSHAPE_A1, -3.0 * (1.0 + (9.0 * p) / 27.0)};
            terms[n++] = (KernelTerm){KSHAPE_C1, -3.0 * (2.0 + (60.0 * p) / 27.0)};
            terms[n++] = (KernelTerm){KSHAPE_B1, 2.0 * 3.0 * (1.0 + (15.0 * p) / 27.0) * (1.0 - t)};
            terms[n++] = (KernelTerm){KSHAPE_B2, 2.0 * 3.0 * (1.0 + (15.0 * p) / 27.0) * t};
            terms[n++] = (KernelTerm){KSHAPE_D1, 3.0 * p / 27.0};
            terms[n++] = (KernelTerm){KSHAPE_E2, -2.0 *3.0 *(20.0 * p) / 27.0};
            terms[n++] = (KernelTerm){KSHAPE_F1, -2.0 * 3.0 * 6.0 * p / 27.0};
            terms[n++] = (KernelTerm){KSHAPE_G1, 2.0 * 3.0 * 15.0 * p / 27.0};
            break;
        }

        default:
            break;
    }
    return n;
}
