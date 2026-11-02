# Kernel_MPI

`Kernel_MPI` computes the primordial non-Gaussianity (f_NL) kernels used to build non-Gaussian
initial conditions. It runs between
two `Zeldovich_MPI` runs:

1. **Dump pass** (`Zeldovich_MPI` built with `DUMP_GAUSSIAN_D`) writes the Gaussian potential as
   y-slices: `Phi_gaussian_y.{y}`.
2. **Kernel pass** (`Kernel_MPI`, this code) reads those slices and writes one set of kernel
   slices for each enabled f_NL shape: `kernel_<shape>_y.{y}`.
3. **Load pass** (`Zeldovich_MPI` built with `LOAD_D_FROM_FILE`) reads the kernel slices and forms
   `phi_G + f_NL * kernel`.

[`1rank-run-cosm500/PPD256_fnl0_cluster.md`](../../1rank-run-cosm500/PPD256_fnl0_cluster.md)
walks through all three passes on a cluster.

## Build

`Kernel_MPI` is built by Meson along with `Zeldovich_MPI` (the target is in
[`src/meson.build`](../meson.build)). It links against `libzeldovich_ic` for the FFT, MPI exchange
and decomposition code.

```bash
meson setup build/kernel                  # once; add --reconfigure if the dir exists
meson compile -C build/kernel Kernel_MPI  # -> build/kernel/src/Kernel_MPI
```

Precision follows `-Ddouble_precision`, which defaults to single. The slice files are float64
either way. A single-precision build converts values on read and write.

## Run

```bash
mpiexec -n R build/kernel/src/Kernel_MPI [param.par] [--png-dir DIR] [--out-dir DIR] [--python-compat]
```

The parameter file defaults to `.runtime/param.par`. It must be a plain `.par` file, because
ParseHeader does not evaluate `#include`, `@Var@` or expressions. Expand a `.par2` file first with
GenParam or `scripts/par2_to_par.py` (see `1rank-run-cosm500/run.sh`).

**Rank count:** `R` must be a multiple of `NumZRanks` and at most `N/2 + 1`. It does not have to
match the dump or load runs, because files are keyed by y only.

### Parameter-file keys

| Key | Required | Meaning |
|---|---|---|
| `BoxSize` | yes | Box size; sets `kfund = 2*pi/BoxSize` |
| `NumZRanks` | yes | z-dimension of the rank grid (same Cartesian topology as `Zeldovich_MPI`) |
| `ZD_f_NL_local`, `ZD_f_NL_equil`, `ZD_f_NL_orthoCMB`, `ZD_f_NL_orthoLSS` | at least one nonzero | Every nonzero shape is computed, one after another |
| `ZD_PNG_DIR` | no (default `png_slices`) | Input directory, also the default output directory |
| `KernelPadM` | no | Padded FFT grid size `M` (see [Padding](#padding)) |

Any other keys are ignored. Only whether an f_NL value is nonzero matters here: its value is
applied later, in the load pass.

### Command-line options and environment variables

Command-line arguments override environment variables, which override the parameter file.

| Option / variable | Effect |
|---|---|
| `PARAM_FILE` | Parameter file, if none is given on the command line |
| `--png-dir DIR` / `PNG_DIR` | Override `ZD_PNG_DIR` |
| `--out-dir DIR` / `OUT_DIR` | Write outputs here instead of the input directory |
| `--python-compat` / `KERNEL_PYTHON_COMPAT=1` | Default `M = 2N`, as in `kernel_MPI.py` |
| `KERNEL_PAD_M=<M>` | Explicit `M`; overrides `KernelPadM` and the default |
| `KERNEL_ZERO_NYQUIST=1` | Zero the `kx`, `ky`, `kz = N/2` modes of phi before computing kernels |
| `KERNEL_ONLY_TERM=<A1\|C1\|...>` | Debug: run one term with coefficient 1 and write `kernel_<term>_y.{y}` |
| `KERNEL_FORCE_STAGED_A2A=1` | Test: force the point-to-point (MPI 3) Alltoallv path instead of `MPI_Alltoallv_c` |

## Inputs and outputs

| | Files | Notes |
|---|---|---|
| In | `<PNG_DIR>/Phi_gaussian_y.{y}`, y = 0 … N/2 | `N` is detected from the size of `Phi_gaussian_y.0` |
| Out | `<OUT_DIR>/kernel_<shape>_y.{y}`, y = 0 … N/2 | `<shape>` is `local`, `equil`, `orthoCMB` or `orthoLSS` |

Each file is one N×N complex slice of interleaved float64 `(real, imag)` values, x fastest and then
z, so element `(x, z)` is at index `z*N + x`. Only the slices with y ≤ N/2 are stored; the other
half follows from Hermitian symmetry.

Rank 0 prints a run summary, the time for each term, a timing breakdown by stage, and peak memory.

## The method

Every term in `kernel_MPI.py` has the form

```
out += coef * J(k) * Q(k),   Q = FFT[ IFFT(G*phi) * IFFT(H*phi) ]
G = |k|^pG,   H = |k|^pH,   J = cJ * |k|^pJ
```

and each f_NL type is a weighted sum of these terms (from [`kernel_terms.c`](kernel_terms.c)):

| f_NL type | Terms |
|---|---|
| local | `A1` |
| equil | `-3 A1 - 6 C1 + 4 B1 + 2 B2` |
| orthoCMB | `-9 A1 - 24 C1 + 10 B1 + 8 B2` |
| orthoLSS | `A1, C1, B1, B2, D1, E2, F1, G1`, with coefficients derived for a scale-invariant spectrum |

The two inverse FFTs share a single complex transform: `z = (G + iH) * phi`, and then
`Q = Im(z^2) / 2`.

### Padding

The quadratic product is computed on a padded `M^3` grid to avoid aliasing. By default `M` is
the smallest valid size of at least `3N/2`, which is alias-free and smaller than the
`M = 2N` used by the Python code. `M` must satisfy `M >= N`, `M % 4 == 0`, and `M - N` even. If
`M < 3N/2` the program warns that the product will alias.

Padding and cropping are never done as separate steps. The original index `i` maps directly to
padded index `pad_idx(i)` (the same mapping as `fftshift -> np.pad -> ifftshift`).

### Parallel layout

There are two decompositions:

- **y-slices (original N grid):** the N/2+1 conjugate pairs `(y, N - y)` are split across ranks.
  Each rank reads, owns and writes its own `Phi_gaussian_y.{y}` / `kernel_<shape>_y.{y}`
  files. Mirror slices are built on the fly from Hermitian symmetry.
- **(x, z) pencils (padded M grid):** each rank owns a block of full-length y columns.

Each term runs in three stages ([`kernel_pass.h`](kernel_pass.h)):

1. **A, forward exchange:** fill `(G + iH) * phi` at the padded positions, run a 2D backward FFT
   on each y-slice, then Alltoallv into pencils.
2. **B, per z-slab:** 1D backward FFT in y, form `Q`, 1D forward FFT in y, then keep
   `y <= N/2`.
3. **C, return exchange:** Alltoallv back to the slice owners, run a 2D forward FFT, crop to N×N,
   then `out += coef * J * Q`.

To save memory, `q_half` reuses the persistent `recv_buffer`, and the shapes are computed one
after another into the same output buffers. Peak memory is therefore the same as for a
single-shape run.

## Source files

| File | Contents |
|---|---|
| [`kernel_main.c`](kernel_main.c) | Driver: arguments, parameters, topology, reading slices, the loop over shapes and terms, writing output, timing |
| [`kernel_params.cpp`](kernel_params.cpp) / `.h` | Reads the `.par` file with ParseHeader |
| [`kernel_grid.c`](kernel_grid.c) / `.h` | Choosing `M`, `pad_idx`/`kfreq`, pair ownership and pencil layout (`KernelLayout`) |
| [`kernel_pass.c`](kernel_pass.c) / `.h` | One kernel term: stages A–C, FFT plans, buffers (`KernelPass`) |
| [`kernel_terms.c`](kernel_terms.c) / `.h` | Shape table (`A1` … `G1`) and the f_NL term coefficients |
| [`kernel_io.c`](kernel_io.c) / `.h` | Slice file read/write, Hermitian fix-ups for the y=0 and y=N/2 planes, Nyquist zeroing |
