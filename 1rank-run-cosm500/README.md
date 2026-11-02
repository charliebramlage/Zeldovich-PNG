# Cosm500 1-rank Zeldovich-MPI run 


## Files

```
Cosm500_1D_1rank/
  cosm500_abacus_legacy_zeldovich.par2   # template (expanded to .runtime/param.par at run time)
  env.sh          # defaults + optional ENV_SCRIPT hook
  build.sh        # meson configure + compile
  run.sh          # wisdom preflight + mpiexec -n 1
  run.batch.example   # copy/edit for PBS or Slurm
  data/           # camb_planck15.dat, eigmodes128
  output/         # default IC output (override with IC_OUTPUT_DIR)
  bin_files/      # FFTW wisdom + rank scratch
  rng_logs/
  .runtime/param.par    # generated .par (GenParam if $ABACUS is set, else scripts/par2_to_par.py)
```

## Prerequisites

- MPI (`mpiexec` or set `MPIEXEC`)
- Meson ≥ 1.7, Ninja
- FFTW with OpenMP (pkg-config `fftw3f`, or set `FFTW_ROOT`)
- Power spectrum file + PLT eigenmodes (see data files below)

## Data files needed to run:

- `data/camb_planck15.dat`
- `data/eigmodes128`

## Run

```bash
# Specify paths
export PK_FILE=/path/to/camb_planck15.dat
export PLT_FILE=/path/to/eigmodes128
export IC_OUTPUT_DIR=/path/to/output # default is `output/`

cd run_bundles/Cosm500_1D_1rank
chmod +x build.sh run.sh
./run.sh
```

## Meson options (`build.sh`)

- `particle_output_mode=3` — CPD-slab streaming (1D layout)
- `use_fftw_wisdom=true`
- `debug_rng_consistency=true`