# Paths are relative to this directory unless prefixed with ../
# Source list follows Makefile groups, sorted per Meson style guide.
# eigmodes128: build/subprojects/zeldovich_mpi/eigmodes128 (Meson copy; legacy zeldovich-PLT uses its own copy).
# =============================================================================
# Zeldovich_MPI — C sources built as C++ (they include C++ headers)
# =============================================================================
ZD_MPI_generation_cpp = configure_file(
  input: 'generation/ZD_MPI_generation.c',
  output: 'ZD_MPI_generation.cxx',
  copy: true,
)
rng_cpp = configure_file(
  input: 'utils/rng.c',
  output: 'rng.cxx',
  copy: true,
)
# =============================================================================
# Zeldovich_MPI — translation units (files())
# =============================================================================
_zd_mpi_core_sources = files(
  '../deps/zeldovich_core/src/parameters.cpp',
  '../deps/zeldovich_core/src/power_spectrum.cpp',
  'communication/mpi_exchange.c',
  'fft/fft_setup.c',
  'fft/fft_wisdom.c',
  'fft/wisdom_rank0.c',
  'ic_stage_api.cpp',
  'zd_log.c',
  'mpi_topology.cpp',
  'output/output_new.cpp',
  'streaming/z_streaming.c',
  'utils/batch_helpers.c',
  'utils/decomposition.c',
  'utils/plt_eigenmodes.c',
  'utils/png_slice_io.c',
  'utils/power_spectrum.c',
  'utils/printing.c',
  'utils/verification.c',
  'utils/zeldovich_wrapper.cpp',
  'zeldovich_mpi_driver.cpp',
)
zd_mpi_sources = _zd_mpi_core_sources
zeldovich_ic_lib_deps = [zd_mpi_common_dep, parseheader_dep]
zeldovich_ic_export_deps = [mpi_dep, omp_dep, fftw_dep, fmt_dep, parseheader_dep]
if not get_option('abacus_embed')
  zd_mpi_sources += files('../deps/zeldovich_core/src/STimer.cc')
endif

zeldovich_ic_lib_sources = zd_mpi_sources + [ZD_MPI_generation_cpp, rng_cpp]
# =============================================================================
# Static library — IC logic + deps (reused by standalone exe and Abacus subproject)
# =============================================================================
zeldovich_ic_lib = static_library(
  'zeldovich_ic',
  zeldovich_ic_lib_sources,
  dependencies: zeldovich_ic_lib_deps,
  install: false,
)
zeldovich_ic_dep = declare_dependency(
  link_with: zeldovich_ic_lib,
  include_directories: zd_mpi_inc,
  dependencies: zeldovich_ic_export_deps,
)
# Standalone tools are not built when embedded in Abacus: libzeldovich_ic omits
# STimer.cc (Abacus multistep provides STimer via proepi.cpp).
if not get_option('abacus_embed')
  executable(
    'Zeldovich_MPI',
    ['main.cpp'],
    link_with: zeldovich_ic_lib,
    dependencies: zd_mpi_common_dep,
    install: true,
  )

  # Kernel_MPI (f_NL kernels, C port of 1rank-run-cosm500/kernel_MPI.py): precision follows
  # -Ddouble_precision like Zeldovich_MPI. The regression test (1e-10 tolerance vs the
  # float64 Python reference) is double only: see the kernel_regression_quick test below.
  kernel_mpi_exe = executable(
    'Kernel_MPI',
    files(
      'kernel/kernel_grid.c',
      'kernel/kernel_io.c',
      'kernel/kernel_main.c',
      'kernel/kernel_params.cpp',
      'kernel/kernel_pass.c',
      'kernel/kernel_terms.c',
    ),
    include_directories: include_directories('kernel'),
    link_with: zeldovich_ic_lib,
    dependencies: [zd_mpi_common_dep, parseheader_dep],
    link_language: 'cpp',
    install: true,
  )
  if _double_prec
    # Regression vs 1rank-run-cosm500/kernel_MPI.py (exit 77 = skipped if pyfftw/mpiexec missing)
    test(
      'kernel_regression_quick',
      find_program('python3'),
      args: [
        meson.project_source_root() / 'tests' / 'kernel_regression' / 'run_regression.py',
        '--tier', 'quick',
        '--kernel-exe', kernel_mpi_exe.full_path(),
        '--work-dir', meson.project_build_root() / 'kernel_regression_work',
      ],
      depends: kernel_mpi_exe,
      is_parallel: false,
      timeout: 1800,
    )
  endif

  # PNG_Combine: white-noise files from Phi_gaussian and the kernels (streaming replacement for the
  # LOAD_D_FROM_FILE pass; same mask, s(k) and arithmetic). Independent of the dump/load defines.
  executable(
    'PNG_Combine',
    files('png_combine/png_combine_main.cpp'),
    include_directories: include_directories('utils'),
    link_with: zeldovich_ic_lib,
    dependencies: [zd_mpi_common_dep, parseheader_dep],
    install: true,
  )

  wisdom_inc = include_directories('.', 'fft')
  executable(
    'wisdom_rank0',
    ['fft/wisdom_rank0.c', 'fft/wisdom_rank0_main.c', 'fft/fft_wisdom.c'],
    include_directories: wisdom_inc,
    dependencies: [fftw_dep, omp_dep, mpi_dep],  # fft_wisdom.c calls MPI_Bcast
    link_language: 'cpp',
    install: true,
  )
endif
