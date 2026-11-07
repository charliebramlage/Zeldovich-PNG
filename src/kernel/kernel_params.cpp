// Kernel_MPI parameter loading via ParseHeader (see kernel_params.h).

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>

#include "ParseHeader.hh"
#include "kernel_params.h"

namespace fs = std::filesystem;

namespace {

// Everything is DONT_CARE: a missing key is reported through found[] so Kernel_MPI can
// MPI_Abort with its own message, instead of ParseHeader calling exit() on rank 0 only.
// NaN marks "not set".
class KernelParameters : public ParseHeader {
public:
    double box_size = std::nan("");
    double num_z_ranks = std::nan("");
    double fnl[4] = {std::nan(""), std::nan(""), std::nan(""), std::nan("")};
    double pad_M = std::nan("");
    std::string png_dir = "png_slices";
    int file_float64 = 0;

    KernelParameters() {
        installscalar("BoxSize", box_size, DONT_CARE);
        installscalar("NumZRanks", num_z_ranks, DONT_CARE);
        installscalar("ZD_f_NL_local", fnl[0], DONT_CARE);
        installscalar("ZD_f_NL_equil", fnl[1], DONT_CARE);
        installscalar("ZD_f_NL_orthoCMB", fnl[2], DONT_CARE);
        installscalar("ZD_f_NL_orthoLSS", fnl[3], DONT_CARE);
        installscalar("KernelPadM", pad_M, DONT_CARE);
        installscalar("ZD_PNG_DIR", png_dir, DONT_CARE);
        installscalar("ZD_PNG_FileFloat64", file_float64, DONT_CARE);
    }
};

void set(double v, double *dst, int *found)
{
    *found = !std::isnan(v);
    *dst = *found ? v : 0.0;
}

}  // namespace

extern "C" int kernel_params_load(const char *path, KernelParams *kp)
{
    if (path == NULL || kp == NULL) return 1;
    std::memset(kp, 0, sizeof(*kp));

    // HeaderStream exits on an unreadable file; check here so the caller can report it.
    FILE *fp = std::fopen(path, "rb");
    if (fp == NULL) return 1;
    std::fclose(fp);

    try {
        KernelParameters p;
        HeaderStream hs{fs::path(path)};
        p.ReadHeader(hs);
        hs.Close();

        set(p.box_size, &kp->box_size, &kp->found[0]);
        set(p.num_z_ranks, &kp->num_z_ranks, &kp->found[1]);
        for (int i = 0; i < 4; i++) set(p.fnl[i], &kp->fnl[i], &kp->found[2 + i]);
        set(p.pad_M, &kp->pad_M, &kp->found[6]);
        if (!kp->found[6]) kp->pad_M = -1.0;
        kp->file_float64 = p.file_float64 ? 1 : 0;

        const std::string dir = p.png_dir.empty() ? std::string("png_slices") : p.png_dir;
        if (dir.size() >= sizeof(kp->png_dir)) return 1;
        std::snprintf(kp->png_dir, sizeof(kp->png_dir), "%s", dir.c_str());
    } catch (...) {
        return 1;
    }
    return 0;
}
