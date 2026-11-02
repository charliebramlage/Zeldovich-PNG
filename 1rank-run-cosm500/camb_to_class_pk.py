"""
Compute the late-time (z=1) and primordial matter power spectra with CLASS,
using the same cosmology as the CAMB run in camb_planck15_params.ini, and
write them out in the same two-column [k, P(k)] format as camb_planck15.dat.

Output files:
    class_pk_z1.dat         -> k [h/Mpc],  P(k,z=1) [(Mpc/h)^3]   (late-time)
    class_pk_primordial.dat -> k [h/Mpc],  P_R(k)   [dimensionless] (primordial)

Note on units/interpretation:
    The late-time file is directly comparable to camb_planck15.dat (matterpower.dat):
    same units, same convention (k in h/Mpc, P(k) in (Mpc/h)^3).

    The primordial file is NOT in the same physical units as the matter power
    spectrum -- P_R(k) is the dimensionless power spectrum of the curvature
    perturbation, P_R(k) = A_s (k/k_pivot)^(n_s-1), before any transfer function
    is applied. It's written with the same k-grid (h/Mpc) purely so the two
    files line up column-for-column for easy plotting/comparison; the second
    column is a different physical quantity, not a matter power spectrum.

Normalization (step 6):
    Also computes the growth ratio D(InitialRedshift)/D(Z_FILE) the way
    Abacus's cosmology module does, and rescales the primordial file so it
    carries the same A_s as the density field the zeldovich code generates.
    If the par file uses ZD_Pk_sigma_ratio, no rescaling is needed (amp = 1).

Usage:
    python camb_to_class_pk.py                  # write both P(k) files
    python camb_to_class_pk.py --write-ratio    # also switch the par file from
                                                # ZD_Pk_sigma to ZD_Pk_sigma_ratio
"""

import argparse
import numpy as np
from scipy.integrate import quad
from classy import Class

ap = argparse.ArgumentParser(description="Write CLASS late-time and primordial P(k) files.")
ap.add_argument("--par", default="cosm500_abacus_legacy_zeldovich.par2",
                help="zeldovich par file to read (and edit with --write-ratio)")
ap.add_argument("--write-ratio", action="store_true",
                help="replace ZD_Pk_sigma with ZD_Pk_sigma_ratio = D(z_init)/D(Z_FILE) in the par file")
args = ap.parse_args()

# ---------------------------------------------------------------------
# 1) Read the k-grid from the CAMB output, so both CLASS files use the
#    exact same k values as camb_planck15.dat for direct comparison.
# ---------------------------------------------------------------------
camb_data = np.loadtxt("data/camb_planck15.dat")
k_hmpc = camb_data[:, 0]          # k in h/Mpc
kmin_hmpc, kmax_hmpc = k_hmpc.min(), k_hmpc.max()

# ---------------------------------------------------------------------
# 1b) Read the normalization inputs from the par file and compute the
#     growth ratio D(z_init)/D(Z_FILE) (see step 6).
# ---------------------------------------------------------------------
PAR_FILE = args.par
Z_FILE = 1.0   # redshift at which the late-time P(k) file is written

def read_par_value(path, key, default=None):
    with open(path) as fp:
        for line in fp:
            line = line.split("#", 1)[0]
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key:
                return float(value.strip().strip('"'))
    if default is not None:
        return default
    raise KeyError(f"{key} not found in {path}")

def abacus_growth_ratio(Omega_M, z1, z2):
    """D(z1)/D(z2) for flat LCDM with no radiation, matching the growth
    function Abacus's cosmology module uses to set ZD_Pk_sigma(_ratio)."""
    E = lambda a: np.sqrt(Omega_M / a**3 + 1.0 - Omega_M)
    D = lambda a: E(a) * quad(lambda x: (x * E(x))**-3, 0.0, a, epsabs=0, epsrel=1e-12)[0]
    return D(1.0 / (1.0 + z1)) / D(1.0 / (1.0 + z2))

def set_par_sigma_ratio(path, ratio):
    """Comment out ZD_Pk_sigma and set ZD_Pk_sigma_ratio = ratio in the par file."""
    with open(path) as fp:
        lines = fp.readlines()
    key_of = lambda line: line.split("#", 1)[0].split("=", 1)[0].strip() if "=" in line.split("#", 1)[0] else None
    new_line = f"ZD_Pk_sigma_ratio = {ratio!r}   # D(z={z_init:g})/D(z={Z_FILE:g}), from camb_to_class_pk.py\n"
    out, placed = [], False
    for line in lines:
        key = key_of(line)
        if key == "ZD_Pk_sigma":
            out.append("# " + line)
            if not placed:
                out.append(new_line)
                placed = True
        elif key == "ZD_Pk_sigma_ratio":
            if not placed:
                out.append(new_line)
                placed = True
        else:
            out.append(line)
    if not placed:
        out.append(new_line)
    with open(path, "w") as fp:
        fp.writelines(out)

z_init = read_par_value(PAR_FILE, "InitialRedshift")
Omega_M = read_par_value(PAR_FILE, "Omega_M")
# abacus_growth_ratio only handles flat LCDM with no smooth component
for key, default in [("w0", -1.0), ("wa", 0.0), ("Omega_K", 0.0), ("Omega_Smooth", 0.0)]:
    if read_par_value(PAR_FILE, key, default) != default:
        raise NotImplementedError(f"{key} != {default} in {PAR_FILE}: growth ratio needs Abacus's full cosmology module")

growth_ratio = abacus_growth_ratio(Omega_M, z_init, Z_FILE)

if args.write_ratio:
    set_par_sigma_ratio(PAR_FILE, growth_ratio)
    print(f"Wrote ZD_Pk_sigma_ratio = {growth_ratio!r} to {PAR_FILE} (ZD_Pk_sigma commented out)")

Pk_sigma = read_par_value(PAR_FILE, "ZD_Pk_sigma", 0.0)
Pk_sigma_ratio = read_par_value(PAR_FILE, "ZD_Pk_sigma_ratio", 0.0)
if (Pk_sigma > 0) == (Pk_sigma_ratio > 0):
    raise ValueError(f"{PAR_FILE} must set exactly one of ZD_Pk_sigma and ZD_Pk_sigma_ratio")

# ---------------------------------------------------------------------
# 2) Map the CAMB ini parameters onto the equivalent CLASS parameters.
#
#    ombh2        = 0.02222   -> omega_b
#    omch2        = 0.1199    -> omega_cdm
#    omnuh2       = 0.0, massive_neutrinos = 0 -> no massive neutrinos
#    massless_neutrinos = 3.04 -> N_ur = 3.04 (CLASS's massless-neutrino species)
#    hubble       = 67.26     -> h = 0.6726
#    temp_cmb     = 2.7255    -> T_cmb
#    helium_fraction = 0.2453 -> YHe
#    scalar_amp(1)        = 2.1e-9  -> A_s
#    scalar_spectral_index(1) = 0.9652 -> n_s
#    scalar_nrun(1)       = 0  -> alpha_s (no running)
#    pivot_scalar = 0.05       -> k_pivot
#    do_nonlinear = 0          -> non linear = none
# ---------------------------------------------------------------------
params = {
    'output': 'mPk',
    'non linear': 'none',
    #'gauge': 'Newtonian', # compute perturbations in Newtonian (longitudinal) gauge
    'gauge' : 'Synchronous',

    'h': 0.6726,
    'omega_b': 0.02222,
    'omega_cdm': 0.1199,
    'N_ur': 3.04,
    'N_ncdm': 0,

    'T_cmb': 2.7255,
    'YHe': 0.2453,

    'A_s': 2.1e-9,
    'n_s': 0.9652,
    'alpha_s': 0.0,
    'k_pivot': 0.05,

    # Make sure CLASS tabulates P(k) over (at least) the full CAMB k-range
    'P_k_max_h/Mpc': kmax_hmpc * 1.05,
    'z_pk': Z_FILE,
}

cosmo = Class()
cosmo.set(params)
cosmo.compute()

h = cosmo.h()

# ---------------------------------------------------------------------
# 3) Late-time matter power spectrum, z = 1
#    classy's cosmo.pk(k, z) expects k in 1/Mpc and returns P(k) in Mpc^3.
#    Convert to the CAMB convention: k in h/Mpc, P(k) in (Mpc/h)^3.
# ---------------------------------------------------------------------
k_1mpc = k_hmpc * h
pk_z1_mpc3 = np.array([cosmo.pk(k, Z_FILE) for k in k_1mpc])   # Mpc^3
pk_z1_hmpc3 = pk_z1_mpc3 * h**3                              # (Mpc/h)^3

late_time_out = np.column_stack([k_hmpc, pk_z1_hmpc3])
np.savetxt(
    "data/class_pk_z1.dat",
    late_time_out,
    fmt="%15.8E",
)

# ---------------------------------------------------------------------
# 4) Primordial (curvature) power spectrum P_R(k)
#    get_primordial() returns CLASS's internal k-grid (1/Mpc) and P_scalar(k).
#    Interpolate (in log-log space, since it's a smooth power law) onto the
#    same k-grid as above so the two files can be compared column-by-column.
# ---------------------------------------------------------------------
prim = cosmo.get_primordial()
k_prim_1mpc = np.array(prim['k [1/Mpc]'])
p_prim = np.array(prim['P_scalar(k)'])

log_p_interp = np.interp(
    np.log(k_1mpc),
    np.log(k_prim_1mpc),
    np.log(p_prim),
)
p_prim_at_k = np.exp(log_p_interp)

# primordial_out = np.column_stack([k_hmpc, p_prim_at_k])
# np.savetxt(
#     "/mnt/user-data/outputs/class_pk_primordial.dat",
#     primordial_out,
#     fmt="%15.8E",
# )

# ---------------------------------------------------------------------
# 5) Convert the dimensionless primordial curvature spectrum P_R(k)
#    (what get_primordial() returns, = A_s (k/k*)^(n_s-1)) into a
#    volume-dimensioned power spectrum, in the SAME units as the
#    late-time file: (Mpc/h)^3, on the same k_h/Mpc grid.
#
#        P_R(k) [Mpc^3]        = (2 pi^2 / k^3) * P_scalar(k)     (k in 1/Mpc)
#        P_R(k_h) [(Mpc/h)^3]  = (2 pi^2 / k_h^3) * P_scalar(k_h*h)
#
#    (the h^3 factors cancel between the Mpc^3 -> (Mpc/h)^3 conversion
#    and the k -> k_h conversion, provided P_scalar is evaluated at the
#    *physical* k = k_h*h, since k_pivot is defined in 1/Mpc)
# ---------------------------------------------------------------------
p_prim_volume_hmpc3 = (2.0 * np.pi**2 / k_hmpc**3) * p_prim_at_k

# ---------------------------------------------------------------------
# 6) Match the primordial amplitude to the sigma8 the code imposes.
#
#    Normalize() in zeldovich_core rescales the late-time file by
#        A = (Pk_sigma / sigma8_file)^2     if ZD_Pk_sigma is set
#        A = Pk_sigma_ratio^2               if ZD_Pk_sigma_ratio is set
#    where sigma8_file is the top-hat sigma(8 Mpc/h) of class_pk_z1.dat.
#    A = g^2 * amp, where g = D(z_init)/D(Z_FILE) is pure growth (must NOT be
#    applied to the primordial spectrum, which does not grow) and amp is the
#    change in A_s relative to this CLASS run's A_s (MUST be applied, so Phi
#    and the density field share the same A_s). g uses Abacus's growth
#    convention, since that is what ZD_Pk_sigma / ZD_Pk_sigma_ratio are built
#    from. The primordial file is loaded with Pk_norm = 0, so amp is baked in here.
# ---------------------------------------------------------------------
# sigma8_file: same integral as PowerSpectrum::sigmaR (k from 0 to 10 h/Mpc),
# on a fine log grid with log-log interpolation of the late-time file.
k_int = np.logspace(-6, 1, 200000)
pk_int = np.exp(np.interp(np.log(k_int), np.log(k_hmpc), np.log(pk_z1_hmpc3)))
x = 8.0 * k_int
w = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
trapezoid = getattr(np, "trapezoid", None) or np.trapz  # renamed in numpy 2.0
sigma8_file = np.sqrt(trapezoid(k_int**3 * w**2 * pk_int / (2.0 * np.pi**2), np.log(k_int)))

if not np.isclose(Omega_M, cosmo.Omega_m(), rtol=1e-4):
    print(f"WARNING: Omega_M in {PAR_FILE} ({Omega_M}) != CLASS Omega_m ({cosmo.Omega_m()}); growth ratio may be off")

if Pk_sigma_ratio > 0:
    # Pure growth back-scaling: the density field keeps this CLASS run's A_s
    if not np.isclose(Pk_sigma_ratio, growth_ratio, rtol=1e-5):
        print(f"WARNING: ZD_Pk_sigma_ratio in {PAR_FILE} ({Pk_sigma_ratio}) != computed "
              f"D(z={z_init:g})/D(z={Z_FILE:g}) ({growth_ratio}); rerun with --write-ratio to update it")
    amp = 1.0
    norm_desc = f"ZD_Pk_sigma_ratio = {Pk_sigma_ratio}"
else:
    amp = (Pk_sigma / (sigma8_file * growth_ratio))**2
    norm_desc = f"ZD_Pk_sigma = {Pk_sigma}"
p_prim_volume_hmpc3 *= amp

print(f"sigma8_file(z={Z_FILE:g})         = {sigma8_file:.6f}")
print(f"D(z={z_init:g})/D(z={Z_FILE:g}) (Abacus) = {growth_ratio:.10f}")
print(f"normalization          : {norm_desc}")
print(f"primordial amp factor  = {amp:.6f}  (sqrt = {np.sqrt(amp):.6f})")
print(f"CLASS sigma8(z=0)      = {cosmo.sigma8():.6f}  -> effective sigma8(z=0) = {cosmo.sigma8() * np.sqrt(amp):.6f}")

primordial_dimensional_out = np.column_stack([k_hmpc, p_prim_volume_hmpc3])
np.savetxt(
    "data/class_pk_primordial_dimensional.dat",
    primordial_dimensional_out,
    fmt="%15.8E",
    header=f"P_zeta(k) [(Mpc/h)^3], rescaled by amp = {amp:.10f} to match {norm_desc} at z = {z_init:g}",
)

# Example: reconstruct delta_prim(k) from a delta_late(k) realization
# delta_prim(k) = sqrt(P_prim(k)/P_late(k,z=1)) * delta_late(k)
# (shown here just as the rescaling factor sqrt(P_prim/P_late), since
#  delta_late(k) itself is whatever complex field/realization you have)
# rescale_factor = np.sqrt(p_prim_volume_hmpc3 / pk_z1_hmpc3)
# np.savetxt(
#     "/mnt/user-data/outputs/class_pk_prim_over_late_rescale_factor.dat",
#     np.column_stack([k_hmpc, rescale_factor]),
#     fmt="%15.8E",
# )

cosmo.struct_cleanup()
cosmo.empty()

print("h =", h)
print("Wrote class_pk_z1.dat and class_pk_primordial.dat")
print("k range:", kmin_hmpc, "to", kmax_hmpc, "h/Mpc")
