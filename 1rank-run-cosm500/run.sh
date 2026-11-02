#!/bin/bash
# Run Cosm500 Zeldovich-MPI on 1 MPI rank.
#
# Usage:
#   ./run.sh              # build, then wisdom preflight + IC generation
#   ./run.sh --no-build   # skip meson compile
#
# Set paths before running (or place files under data/):
#   export PK_FILE=/path/to/camb_planck15.dat
#   export PLT_FILE=/path/to/eigmodes128
#   export PRIMORDIAL_PK_FILE=/path/to/class_pk_primordial_dimensional.dat  # optional
#   export IC_OUTPUT_DIR=/path/to/output   # default: ./output
#   export NTHREADS_PER_RANK=8
#   export MPIEXEC_EXTRA="--bind-to core"  # optional
#   export ABACUS=/path/to/abacus-public   # optional: use Abacus's GenParam for .par2 -> .par
#
# As in Abacus, the .par2 template is expanded into a plain .par (.runtime/param.par) and
# the executables read that with ParseHeader. The expansion uses $ABACUS/Abacus/GenParam.py
# when ABACUS is set, otherwise ../scripts/par2_to_par.py (same rules, no Abacus needed).
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${BUNDLE_DIR}/../.." && pwd)}"
MESON_BUILD="${MESON_BUILD:-${REPO_ROOT}/build/cosm500_1d_1rank}"

TEMPLATE_PAR2="${BUNDLE_DIR}/cosm500_abacus_legacy_zeldovich.par2"
RUNTIME_DIR="${BUNDLE_DIR}/.runtime"
PARAM_FILE="${RUNTIME_DIR}/param.par"
PAR2_TO_PAR="${BUNDLE_DIR}/../scripts/par2_to_par.py"
BIN_DIR="${BUNDLE_DIR}/bin_files"
RNG_LOG_DIR="${BUNDLE_DIR}/rng_logs"
WISDOM_LOG="${BUNDLE_DIR}/wisdom_preflight.log"
RUN_LOG="${BUNDLE_DIR}/cosm500_abacus_legacy_zeldovich_bin_generation.log"

# qdensity+PLT => narray=4 (see zeldovich_mpi_driver.cpp)
NARRAY=4

DO_BUILD=1
if [[ "${1:-}" == "--no-build" ]]; then
  DO_BUILD=0
fi

# shellcheck source=/dev/null
source "${BUNDLE_DIR}/env.sh"

# Optional: primordial P(k) file. Unlike PK_FILE/PLT_FILE, this feature degrades
# gracefully if missing (the driver just disables the primordial lookup with a
# warning, rather than aborting) -- so default it here rather than requiring
# env.sh to always set it, and don't hard-fail if it's absent.
PRIMORDIAL_PK_FILE="${PRIMORDIAL_PK_FILE:-${DATA_DIR}/class_pk_primordial_dimensional.dat}"
PNG_DIR="${PNG_DIR:-${BUNDLE_DIR}/png_slices}"

if [[ ! -f "${PK_FILE}" ]]; then
  echo "ERROR: PK file not found: ${PK_FILE}" >&2
  echo "  Set PK_FILE or place camb_planck15.dat under ${DATA_DIR}/" >&2
  exit 1
fi
if [[ ! -f "${PLT_FILE}" ]]; then
  echo "ERROR: PLT eigenmode file not found: ${PLT_FILE}" >&2
  echo "  Set PLT_FILE or place eigmodes128 under ${DATA_DIR}/" >&2
  exit 1
fi
if [[ ! -f "${PRIMORDIAL_PK_FILE}" ]]; then
  echo "WARNING: Primordial PK file not found: ${PRIMORDIAL_PK_FILE}" >&2
  echo "  (optional feature; run will proceed with the primordial lookup disabled)" >&2
  echo "  Set PRIMORDIAL_PK_FILE or place class_pk_primordial_dimensional.dat under ${DATA_DIR}/" >&2
fi

mkdir -p "${RUNTIME_DIR}" "${IC_OUTPUT_DIR}" "${BIN_DIR}" "${RNG_LOG_DIR}" "${PNG_DIR}"

# Expand the .par2 template into the runtime .par, overriding the paths with resolved
# absolute ones (overrides win over values in the .par2, as in GenParam.makeInput).
rm -f "${PARAM_FILE}"
if [[ -n "${ABACUS:-}" && -f "${ABACUS}/Abacus/GenParam.py" ]]; then
  echo "par2 -> par: GenParam (${ABACUS}/Abacus/GenParam.py)"
  if ! PYTHONPATH="${ABACUS}${PYTHONPATH:+:${PYTHONPATH}}" python3 - "${TEMPLATE_PAR2}" "${PARAM_FILE}" <<PYEOF
import sys
from Abacus.GenParam import makeInput
makeInput(sys.argv[2], sys.argv[1],
          InitialConditionsDirectory="${IC_OUTPUT_DIR}",
          ZD_Pk_filename="${PK_FILE}",
          ZD_PLT_filename="${PLT_FILE}",
          ZD_Pk_primordial_filename="${PRIMORDIAL_PK_FILE}",
          ZD_PNG_DIR="${PNG_DIR}")
PYEOF
  then
    echo "WARNING: GenParam failed; falling back to scripts/par2_to_par.py" >&2
    rm -f "${PARAM_FILE}"
  fi
fi
if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "par2 -> par: ${PAR2_TO_PAR}"
  python3 "${PAR2_TO_PAR}" "${TEMPLATE_PAR2}" "${PARAM_FILE}" \
    -s "InitialConditionsDirectory=${IC_OUTPUT_DIR}" \
    -s "ZD_Pk_filename=${PK_FILE}" \
    -s "ZD_PLT_filename=${PLT_FILE}" \
    -s "ZD_Pk_primordial_filename=${PRIMORDIAL_PK_FILE}" \
    -s "ZD_PNG_DIR=${PNG_DIR}"
fi

if (( DO_BUILD )); then
  # shellcheck source=/dev/null
  source "${BUNDLE_DIR}/build.sh"
else
  export ZMPI_EXE="${MESON_BUILD}/src/Zeldovich_MPI"
  export WISDOM_EXE="${MESON_BUILD}/src/wisdom_rank0"
fi

if [[ ! -x "${ZMPI_EXE}" || ! -x "${WISDOM_EXE}" ]]; then
  echo "ERROR: executables missing; run ./build.sh first" >&2
  exit 1
fi

# N for the wisdom preflight (should match NP in the .par)
N=128

if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE}" ]]; then
  NTOTRANKS="$(wc -l < "${PBS_NODEFILE}")"
elif [[ -n "${SLURM_NTASKS:-}" ]]; then
  NTOTRANKS="${SLURM_NTASKS}"
else
  NTOTRANKS=1
fi

export OMP_NUM_THREADS="${NTHREADS_PER_RANK}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

rm -rf "${BIN_DIR}"/rank_* 2>/dev/null || true

echo "Wisdom preflight: ${WISDOM_EXE} ${N} ${NARRAY} (cwd=${BIN_DIR})"
cd "${BIN_DIR}"
"${WISDOM_EXE}" "${N}" "${NARRAY}" >> "${WISDOM_LOG}" 2>&1
if [[ ! -f "fftw_wisdom.wisdom" ]]; then
  echo "ERROR: wisdom preflight did not create fftw_wisdom.wisdom in ${BIN_DIR}" >&2
  exit 1
fi

echo "Running: ${MPIEXEC} -n ${NTOTRANKS} ${MPIEXEC_EXTRA} ${ZMPI_EXE} ${PARAM_FILE}"
ulimit -c unlimited

# shellcheck disable=SC2086
${MPIEXEC} -n "${NTOTRANKS}" ${MPIEXEC_EXTRA} \
  "${ZMPI_EXE}" "${PARAM_FILE}" \
  > "${RUN_LOG}" 2> "${RNG_LOG_DIR}/zd_mpi_rng_debug.log"

grep "\[REAL-FFT-DEBUG\]" "${RNG_LOG_DIR}/zd_mpi_rng_debug.log" \
  > "${RNG_LOG_DIR}/zd_mpi_real_fft_debug.txt" 2>/dev/null || true

IC_COUNT="$(find "${IC_OUTPUT_DIR}" -type f \( -name 'ic_*' -o -path '*/ic/*' \) 2>/dev/null | wc -l)"
echo "Done. Particle IC segments under ${IC_OUTPUT_DIR}: ${IC_COUNT} files"
echo "  param: ${PARAM_FILE}"
echo "  kernel: ${MPIEXEC} -n <R> ${MESON_BUILD}/src/Kernel_MPI ${PARAM_FILE}"
echo "  stdout: ${RUN_LOG}"
echo "  stderr: ${RNG_LOG_DIR}/zd_mpi_rng_debug.log"

if [[ "${IC_COUNT}" -eq 0 ]]; then
  echo "WARNING: no particle IC files found under ${IC_OUTPUT_DIR}" >&2
  exit 1
fi
