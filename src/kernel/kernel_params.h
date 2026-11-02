#ifndef KERNEL_PARAMS_H
#define KERNEL_PARAMS_H

// Kernel_MPI parameters, read from a plain .par file with ParseHeader -- the same parser
// Zeldovich_MPI and Abacus use. Expand .par2 files to .par first (GenParam, or
// scripts/par2_to_par.py); ParseHeader does not evaluate #include, @Var@ or expressions.

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double box_size;
    double num_z_ranks;
    double fnl[4];       // local, equil, orthoCMB, orthoLSS
    double pad_M;        // KernelPadM, or -1
    int found[7];        // BoxSize, NumZRanks, 4 x f_NL, KernelPadM
    char png_dir[4096];  // ZD_PNG_DIR, or "png_slices"
} KernelParams;

// Keys: BoxSize, NumZRanks, ZD_f_NL_local, ZD_f_NL_equil, ZD_f_NL_orthoCMB,
// ZD_f_NL_orthoLSS, optional ZD_PNG_DIR (default "png_slices"), optional KernelPadM.
// Other keys in the file are ignored. Returns 0 on success, non-zero if the file cannot
// be read or ZD_PNG_DIR does not fit. Missing keys are reported through found[], not here.
// A syntax error in the file makes ParseHeader print the error and exit.
int kernel_params_load(const char *path, KernelParams *kp);

#ifdef __cplusplus
}
#endif

#endif // KERNEL_PARAMS_H
