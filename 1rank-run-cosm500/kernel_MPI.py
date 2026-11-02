import numpy as np
import pyfftw
import os
import time
t_start = time.perf_counter()

# Steps for MPI (from zeldovich_mpi_driver.cpp) 
# * 7. MULTI-BATCH LOOP (for each batch) 
#  *    a. Get batch’s (y_primary, y_mirror)
#  *    b. Alloc local_y_slices + fft_stage_2d → generate + 2D FFT → free stage
#  *    c. calculate_batch_send_recv_counts; pack_slices_to_send_buffer; free local_y_slices
#  *    d. MPI_Alltoallv_c(send_buffer -> recv_buffer)
#  *    e. Update src_write_cursor; free per-batch send buffer

# Tell pyfftw to use the same number of threads as your system
pyfftw.interfaces.cache.enable()

# File format for every y-slice file (no x,y,z indices stored): N*N complex values, each stored as
# interleaved float64 (real, imag), with x the fastest index and then z. So element (x, z) is at
# position z*N + x, and a slice reshapes to (N, N) as [z, x].
file_dtype = np.dtype(np.complex128)
 

#parse parameter file for values
def parse_par2(path):
    """Parse a .par2 file into a dict of {key: value}.
    Handles 'key = value' lines, trailing/full-line '#' comments, and quoted
    strings. Values that look numeric are returned as float; otherwise as
    stripped strings.
    """
    params = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()  # drop inline/full-line comments
            if not line or '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            try:
                val = float(val)
            except ValueError:
                pass
            params[key] = val
    return params

# ---------------- y-pair ownership (same formula as zeldovich_mpi_driver.cpp / batch_helpers.c) ----
def rank_pair_range(r, total_pairs, num_ranks): #returns the y_primary values owned by rank "r" when total_pairs pairs are split over num_ranks ranks
    per, rem = divmod(total_pairs, num_ranks)
    num_pairs = per + 1 if r < rem else per
    start = r * (per + 1) if r < rem else rem * (per + 1) + (r - rem) * per
    return list(range(start, start + num_pairs))

#Get the BoxSize (L), NumZRanks, and the shared slice directory from the same param.par
#that run.sh feeds to Zeldovich_MPI.
PARAM_FILE = os.environ.get("PARAM_FILE", ".runtime/param.par")
par2_params = parse_par2(PARAM_FILE)
L = par2_params["BoxSize"] ##think about units
PNG_DIR = par2_params.get("ZD_PNG_DIR", "png_slices")
num_z_ranks = int(par2_params["NumZRanks"])   # Abacus NumZRanks = size_z of the rank grid

# kernel() below is a serial emulation of the MPI pencil exchange. num_ranks only
# sets that fake grid. Slice files are keyed by y, not by rank
# (ZD_PNG_DIR/Phi_gaussian_y.{y} and kernel_y.{y}), so a later C load can use a
# different rank count. Default the emulation to NumZRanks (one x-column).
num_ranks = int(os.environ.get("NUM_RANKS", num_z_ranks))
assert num_ranks % num_z_ranks == 0, f"num_ranks={num_ranks} is not divisible by NumZRanks={num_z_ranks}"

#Obtain the PPD from data. Planes are keyed by y only: png_slices/Phi_gaussian_y.<y>
sample_slice = np.fromfile(os.path.join(PNG_DIR, "Phi_gaussian_y.0"), dtype=file_dtype)
N = int(round(np.sqrt(sample_slice.shape[0])))
assert N * N == sample_slice.shape[0], "Phi_gaussian_y.0 does not hold N*N complex values"
print("N", N)
print(f"kernel emulation ranks {num_ranks} (size_x={num_ranks // num_z_ranks} x size_z={num_z_ranks})")


# Create aligned arrays for phi_gaussian(k) and |k|
phi_k = pyfftw.empty_aligned((N, N, N), dtype='complex128')
k_arr=pyfftw.empty_aligned((N,N,N), dtype='float64')

#upload data from the dumped files into phi_k
for y in range(N//2 +1):
    filename = os.path.join(PNG_DIR, f"Phi_gaussian_y.{y}")
    slice_zx = np.fromfile(filename, dtype=file_dtype)
    assert slice_zx.shape[0] == N*N, f"{filename}: expected {N*N} values, got {slice_zx.shape[0]}"
    phi_k[:, y, :] = slice_zx.reshape(N, N).T   # file is [z][x] (x fastest); phi_k is [x][y][z]
kfreq_1d = np.fft.fftfreq(N) * N          # [0, 1, ..., N/2-1, -N/2, ..., -1]
KX, KY, KZ = np.meshgrid(kfreq_1d, kfreq_1d, kfreq_1d, indexing='ij')
k_arr = ((2*np.pi) / L) * np.sqrt(KX**2 + KY**2 + KZ**2) 


##Handling Hermitian Symmetry##
index=np.arange(N) ##[ 0 1 2...N-1]
mirror= (-index) % N ##[0 N-1 N-2 ... 1]

y_index = np.arange(1,N//2) ##[1 2 ... N/2-1] excludes 250 if N=500
y_mirror = N-y_index ##[N-1 N-2 ... N/2 +1] excludes 250

phi_k[np.ix_(mirror,y_mirror,mirror)] = np.conj(phi_k[np.ix_(index, y_index, index)]) #This is Hermitian symmetry, it matches negative k values (indexed as -x --> N-x) to the conjugate of the positive k values

##Self-conjugate planes: y=0 and y=Nhalf need internal mirroring
for y0 in (0, N//2):
   z_index = np.arange(1, N//2)
   phi_k[np.ix_(mirror, [y0], N - z_index)] = np.conj(phi_k[np.ix_(index, [y0], z_index)]) #same mirroring process but excluding z=0 and z=N//2
   x_index = np.arange(1, N//2)
   phi_k[N - x_index, y0, 0]      = np.conj(phi_k[x_index, y0, 0]) #handling z=0 and z=N//2 cases (excluding x=0 and x=N//2)
   phi_k[N - x_index, y0, N//2]  = np.conj(phi_k[x_index, y0, N//2])
   for xx in (0, N//2):
       for zz in (0, N//2):
           phi_k[xx, y0, zz] = phi_k[xx, y0, zz].real  # force real at DC/Nyquist corners 

#In C, here we should free the index, mirror, y_index and y_mirror arrays



##Formulas for padding and cropping arrays (which we do to prevent aliasing)##
def pad_kspace(array, N, N2):
    pl = (N2 - N) // 2 # calculates how many zeros to pad on the left side of the array to get the array from N to N2
    pr = (N2 - N) - pl # calculates how many zeros to pad on the right side of the array
    out = np.fft.fftshift(array, axes=(0, 1, 2)) # shifts the array so that the back half of the data is now in the beginning entries (aka shifts the 0 mode to the center of the array) e.g. [0,1,2,3] --> [2,3,0,1]
    out = np.pad(out, ((pl, pr), (pl, pr), (pl, pr))) # pads the array with zeros on the left and right side of each axis
    return np.fft.ifftshift(out, axes=(0, 1, 2)) # shifts the array back so the padded zeros are now in the center

def crop_kspace(array_padded, N, N2):
    start = (N2 - N) // 2 # calculates how many zeros have been added to the left side get the original array from size N to size N2
    out = np.fft.fftshift(array_padded, axes=(0, 1, 2)) #shifts the array. Ensures the padded zeros are on the left and right ends of the array
    out = out[start:start+N, start:start+N, start:start+N] #only selects the center values
    return np.fft.ifftshift(out, axes=(0, 1, 2)) # shifts the array back to normal


##pad phi_k
phi_k_pad_data=pad_kspace(phi_k, N, 2*N) #created the padded phi array
#In C, here we should free phi_k (unpadded at this point) (not used again)
phi_k_pad=pyfftw.empty_aligned((2*N, 2*N, 2*N), dtype='complex128')
phi_k_pad[:]=phi_k_pad_data #save padded phi_k into the pyfftw aligned empty array. Have to do it in these two steps to ensure the padded array is aligned for pyfftw.
#In C, free phi_k_pad_data (not used again)


# ============================================================================
# Serial emulation of the repo's design (pack in mpi_exchange.c, unpack in
# z_streaming.c) applied to the kernel
# "Ranks" are just list entries; every "send/recv" is a copy.
# ============================================================================
M = 2*N                       # stands in for 2N (padded size)
# rank grid = the grid of the C run (zeldovich_mpi_driver.cpp: dims = {num_ranks/NumZRanks, NumZRanks});
# the same ranks hold y-slices AND pencils in different stages
size_z = num_z_ranks
size_x = num_ranks // size_z
PACK_GH = True               # repo's "D + i*F" trick: carry G*phi and H*phi in ONE complex array

# ---------------- decomposition helpers (mirror decomposition.c / batch_helpers.c) -------------
def split(n, parts, i): #tells you the start and end values of the i-th piece of #n items split into #parts
    base, rem = divmod(n, parts)
    start = i * (base + 1) if i < rem else (rem * (base + 1)) + ((i - rem) * base)
    return start, start + (base + 1 if i < rem else base)

def grid_bounds(dest):                 #converts the rank # into it's x bounds and z bounds      # dest = rank_x*size_z + rank_z #has z as the fastest changing index
    rx, rz = divmod(dest, size_z)
    return (*split(M, size_x, rx), *split(M, size_z, rz))   # x0, x1, z0, z1

total_pairs = N + 1                          # M/2 + 1 pairs on the padded grid
def rank_pairs(r):                           # returns the padded-grid y_primary values owned by rank "r" (one pair of y-values per batch)
    return rank_pair_range(r, total_pairs, num_ranks)

def batch_y(r, b):                           # returns the y values rank "r" processes in batch b (one pair of y-values per batch)
    p = rank_pairs(r) #list of y_primary values owned by rank "r"
    if b >= len(p): #if the batch number is greater than the number of y_primary values owned by rank "r", return an empty list
        return []
    yp = p[b] #return the y_primary value for the batch number b
    ym = yp if yp in (0, N) else M - yp #y_mirror value for the y_primary value yp, if yp is not 0 or M/2, return M - yp
    return [yp] if yp == ym else [yp, ym]

num_batches = max(len(rank_pairs(r)) for r in range(num_ranks)) #number of batches is the maximum number of y_primary values owned by any rank

# ---------------- FFTW PLANNING: small STAGING buffers, planned once, kept for the whole run ----
stage2d = pyfftw.empty_aligned((M, M), dtype='complex128')   # one x-z slice
buf1d = pyfftw.empty_aligned(M, dtype='complex128')          # one y pencil (one per OMP thread in C)
fl = ('FFTW_MEASURE',)
plan2d_bwd = pyfftw.FFTW(stage2d, stage2d, axes=(0, 1), direction='FFTW_BACKWARD', flags=fl)
plan2d_fwd = pyfftw.FFTW(stage2d, stage2d, axes=(0, 1), direction='FFTW_FORWARD',  flags=fl)
plan1d_bwd = pyfftw.FFTW(buf1d, buf1d, axes=(0,), direction='FFTW_BACKWARD', flags=fl)
plan1d_fwd = pyfftw.FFTW(buf1d, buf1d, axes=(0,), direction='FFTW_FORWARD',  flags=fl)

# NOTE: normalise_idft=False goes on BACKWARD calls only. On a FORWARD plan, pyfftw (see
# pyfftw.pyx ~line 1854) treats normalise_idft=False as "scale the forward by 1/N" -- so never pass it there.
def run2d(plan, arr2d, backward):    # staged: copy in -> execute on the planned buffer -> copy out
    stage2d[:] = arr2d
    plan(normalise_idft=False) if backward else plan()
    arr2d[:] = stage2d

def run1d(plan, line, backward):
    buf1d[:] = line
    plan(normalise_idft=False) if backward else plan()
    line[:] = buf1d



##define function for convolution
#before using kernel function: use pyfftw empty for G, H and define G, H for your specific kernel
def kernel(G,H): #inputs G(k_1) and H(k_1) then returns your Q(k), which must then be multiplied by J(k) to get the final kernel 
    #pad the G and H arrays
    G_pad_data=pad_kspace(G, N, 2*N)
    #In C, free G
    G_pad=pyfftw.empty_aligned((2*N, 2*N, 2*N), dtype='complex128')
    G_pad[:]=G_pad_data
    #In C, free G_pad_data (not used again)
    H_pad_data=pad_kspace(H, N, 2*N)
    #In C, free H
    H_pad=pyfftw.empty_aligned((2*N, 2*N, 2*N), dtype='complex128')
    H_pad[:]=H_pad_data
    #In C, free H_pad_data (not used again)
    #Here, we're at 4 arrays in memory: phi_k_pad, k_array, G_pad, and H_pad

    Gphi_k = pyfftw.empty_aligned((2*N, 2*N, 2*N), dtype='complex128')
    Gphi_k=np.multiply(phi_k_pad,G_pad) #Multiply phi(k_1)*G(k_1)
    #In C, free G_pad


    Hphi_k = pyfftw.empty_aligned((2*N, 2*N, 2*N), dtype='complex128')
    Hphi_k=np.multiply(phi_k_pad, H_pad) #Multiply phi(k_2)*H(k_2)

    inputs = [Gphi_k + 1j * Hphi_k] if PACK_GH else [Gphi_k, Hphi_k] 
    narray = len(inputs)         # 1 if packed, else 2
    #In C, free H_pad

    # ======================= STAGE A: y-slices -> 2D bwd FFT -> pack -> Alltoallv, per batch ===========
    # recv_chunks[dest][src] = list over batches of 1D arrays (what src sends to dest in that batch) (each batch gives a 1D array)
    recv_chunks = [[[] for _ in range(num_ranks)] for _ in range(num_ranks)] #creates empty list of lists thats (x_size) x (y_size)
    #runs the 2D FFT on each y-slice for each rank, batch by batch, and packs the data into the recv_chunks list for later unpacking
    for b in range(num_batches):
        for src in range(num_ranks):
            ys = batch_y(src, b) #y-values that rank "src" processes in batch "b"
            ns = len(ys) #number of y-slices in this batch for this rank (1 or 2)
            # local_y_slices layout: [slice][array][x][z]   (Y_SLICE macro) #array is 0 if packed, else it's 0 Or 1 for G or H
            local = np.zeros((ns, narray, M, M), dtype=complex)
            for s, y in enumerate(ys):
                for a in range(narray):
                    local[s, a] = inputs[a][:, y, :] #fills local[s,a] with the y-slice of the input array (G or H) for the given y-value
                    run2d(plan2d_bwd, local[s, a], True) #runs the 2D FFT on the y-slice (taking it to real space)
            for dest in range(num_ranks):                       # pack_slices_to_send_buffer
                x0, x1, z0, z1 = grid_bounds(dest)
                chunk = local[:, :, x0:x1, z0:z1]               # [slice][array][x][z]
                packed = np.ascontiguousarray(chunk.transpose(1, 0, 2, 3)).ravel()   # [array][slice][lx][lz]  #copies the array into a contiguous block of memory and flattens it into a 1D array
                recv_chunks[dest][src].append(packed)           # "Alltoallv": src's segment lands at dest #puts the packed data into the recv_chunks list, labeled by the destination rank and the source rank, for later unpacking
                                                                #Here, send and recv are the same, because we're just copying data around in memory

    src_batch_slice_counts = [[len(batch_y(s, b)) for b in range(num_batches)] for s in range(num_ranks)]
    y_owner_src = np.zeros(M, int); y_batch_idx = np.zeros(M, int); y_slice_idx = np.zeros(M, int)
    for s in range(num_ranks):
        for b in range(num_batches):
            for j, y in enumerate(batch_y(s, b)):
                y_owner_src[y], y_batch_idx[y], y_slice_idx[y] = s, b, j #creates a mapping of each y-value to its source rank, batch index, and slice index #slice index tells you if the y-slice is y_primary or y_mirror (0 or 1)
    assert sorted(y for r in range(num_ranks) for b in range(num_batches) for y in batch_y(r, b)) == list(range(M))   # checks that every y covered once

    # recv_buffer[dest] = [src][batch][array][slice][pencil], plus displacement per source rank (saved in recv_displs_src for unpacking)
    #packs all the data for each destination rank into a single 1D array
    recv_buffer, recv_displs_src = [], []
    for dest in range(num_ranks):
        parts = [np.concatenate(recv_chunks[dest][s]) if recv_chunks[dest][s] else np.empty(0, complex) #strings all chunks for a given destination rank into a single array
                for s in range(num_ranks)]
        displs = np.cumsum([0] + [len(p) for p in parts[:-1]]) #gives the start position of each source rank's data in the concatenated buffer
        recv_buffer.append(np.concatenate(parts)) #strings all the parts for a destination into a single 1D array, then appends that to the array for ALL destination ranks 
                                                  #recv_buffer is a list with one entry per destination rank
        recv_displs_src.append(displs) #saves the displacements for each source rank's data in the concatenated buffer for the destination rank
   

   
    # ======================= STAGE B: per z-slab unpack -> bwd 1D -> multiply -> fwd 1D ================
    q_pencils = []                                              # [dest] -> (pencil_count, M)  y contiguous
    for dest in range(num_ranks):
        x0, x1, z0, z1 = grid_bounds(dest)
        x_count, z_count = x1 - x0, z1 - z0
        pencil_count = x_count * z_count
        qp = np.zeros((pencil_count, M), dtype=complex)
        for z_global in range(z0, z1):                          # z_streaming_unpack, one z at a time, gathering all pencils for that z-slab 
            z_idx = z_global - z0
            zslab = np.zeros((narray, x_count, M), dtype=complex)       # ZSLAB [array][x][y], y fastest
            #get the correct index for data, then fill in the zslab with the data from the recv_buffer for the destination rank
            for a in range(narray):
                for x_idx in range(x_count):
                    for y in range(M):
                        src, batch, sl = y_owner_src[y], y_batch_idx[y], y_slice_idx[y] #calling the source rank, batch, and slice index for the y-value (from the mapping created earlier)
                        pencil_idx = x_idx * z_count + z_idx
                        batch_offset = sum(src_batch_slice_counts[src][bb] * pencil_count * narray
                                        for bb in range(batch))  ##gives the offset for the current batch in the recv_buffer for the source rank
                        num_sl = src_batch_slice_counts[src][batch]
                        off = (recv_displs_src[dest][src] + batch_offset #recv_displs_src[dest][src] gets us the starting index of the data for the source rank in the recv_buffer for the destination rank, batch_offset gets us to the start of the current batch
                            + a * num_sl * pencil_count + sl * pencil_count + pencil_idx) #then a * num_sl * pencil_count (gets you to the correct array value), sl * pencil_count (gets you to the correct slice), and pencil_idx (gets you to the correct pencil within that slice)
                        zslab[a, x_idx, y] = recv_buffer[dest][off] ##fills in the data for this specific index in zslab with the correct data. Loops until z slab is filled for all arrays, x indices, and y values
            for a in range(narray):                             # backward 1D FFT, staged per line (taking us to real space)
                #doing the 1D transform for each x_index
                for x_idx in range(x_count):
                    run1d(plan1d_bwd, zslab[a, x_idx], True)
            #multiply Gphi(x)*Hphi(x)=Q(x), (now narray, a, is irrelevant because we've combined g and h into Q)
            if PACK_GH:                                         # z = g + i h with g,h real  =>  g*h = Im(z^2)/2
                Qx = ((zslab[0] ** 2).imag / 2 / M**3).astype(complex)
            else:
                Qx = zslab[0] * zslab[1] / M**3                  # multiply + the single 1/M^3 (local, no MPI)
            # forward 1D FFT, staged per line (taking us to frequency space)
            for x_idx in range(x_count):                        
                run1d(plan1d_fwd, Qx[x_idx], False)
            for x_idx in range(x_count):
                qp[x_idx * z_count + z_idx, :] = Qx[x_idx] #by the end of the loops, qp will contain all the pencils (number: x_count x z_count) for the destination rank (with each pencil being a 1D array of length M (the y-dimension))
        q_pencils.append(qp) ##q_pencils[dest][pencil_idx, ky], qp_pencils is a list of arrays, one for each destination rank, where each array is 2D with shape (pencil_count, M) 

    # ======================= RETURN TRIP: pencils -> y-slices -> 2D fwd FFT ===========================
    
    #rebuild y-slices from pencils, then run 2D FFT on each y-slice, then slot into the correct y-slice in Qk
    yslices_of = [[y for b in range(num_batches) for y in batch_y(r, b)] for r in range(num_ranks)] 
    Qk = np.zeros((M, M, M), dtype=complex)                     # reassembled [x][y][z] for checking
    for r in range(num_ranks): #each rank r will receive the pencils from all other ranks and then control a group of y-slices
        out = np.zeros((len(yslices_of[r]), M, M), dtype=complex)    # [slice][x][z]
        for d in range(num_ranks):                              # pack at d, "send" to r, unpack at r
            x0, x1, z0, z1 = grid_bounds(d)
            blk = q_pencils[d][:, yslices_of[r]]                     # (pencils, nslices) #from pencils owned destination rank d, (using every pencil in d) it picks out the y-values (of the y-slices) owned by rank r
            blk = np.ascontiguousarray(blk.T).reshape(len(yslices_of[r]), x1 - x0, z1 - z0)   # [slice][lx][lz] #.T makes the y-slices the first dimension, then reshapes it to have the correct x and z dimensions
            out[:, x0:x1, z0:z1] = blk #selects the rectangle rank d owns, then slots the data from blk into the correct position. fills in the x0-x1 x z0-z1 rectangle of each y-slice (in rank r, with the data obtained from rank d's pencils)
                                    #out gets rewritten every r loop. After all d loops, out will contain all the y-slices (with full xz planes) for rank r      
        #2D FFT on each y-slice in rank r, then slot into the correct y-slice in Qk
        for s, y in enumerate(yslices_of[r]):
            run2d(plan2d_fwd, out[s], False) #runs the 2D FFT on each y-slice in rank r, taking it back to frequency space
            Qk[:, y, :] = out[s] #slots the  transformed y-slice into the correct y-slice in Qk
            #Qk is the final (padded) output array that contains all of the data from all ranks

    Qk[0][0][0] = 0   ##Set DC mode as zero
    Q_k_data=crop_kspace(np.asarray(Qk),N,2*N)
    #In C, free Qk
    Q_k= pyfftw.empty_aligned((N, N, N), dtype='complex128')
    Q_k[:]=Q_k_data[:]
    #In C, free Q_k_data

    #3 arrays: phi_k_pad, k_arr, Q_k
    #phi_k_pad and k_arr are always around (must be used for all kernels), so after running a kernel function, we've only added one total array

    return Q_k


#Select which f_NL to calculate
local = par2_params["ZD_f_NL_local"] 
equil = par2_params["ZD_f_NL_equil"]
orthoCMB = par2_params["ZD_f_NL_orthoCMB"]
orthoLSS = par2_params["ZD_f_NL_orthoLSS"]


#f_NL local
if(local!=0):
    G_A1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    H_A1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    G_A1[:]=1
    H_A1[:]=1
    Q_A1=kernel(G_A1, H_A1)
    J_A1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    J_A1[:]=1
    K_A1=np.multiply(Q_A1,J_A1)
    #In C, free Q_A1, J_A1

    kernel_output=K_A1
    #In C, free K_A1


#input f_NL equil and f_NL orthoCMB (which use the same kernels)
if(equil!=0 or orthoCMB!=0):

    #Kernel K_A^I
    G_A1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    H_A1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    G_A1[:]=1
    H_A1[:]=1
    Q_A1=kernel(G_A1, H_A1)
    J_A1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    J_A1[:]=1
    K_A1=np.multiply(Q_A1,J_A1)
    #In C, free Q_A1, J_A1
    if(equil!=0):
        kernel_equil=-3*K_A1
    if(orthoCMB!=0):
        kernel_orthoCMB=-9*K_A1
    #In C, free K_A1


    #Kernel K_C^I
    G_C1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    H_C1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    G_C1[:]=k_arr
    H_C1[:]=k_arr
    Q_C1=kernel(G_C1, H_C1)

    k_arr[0][0][0]=1 #momentarily modify the k_arr to avoid dividing by zero, which won't matter because we will set [0][0][0]=0 for the DC mode anyway
    J_C1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    J_C1=(1/3)*(1/np.multiply(k_arr,k_arr))
    J_C1[0][0][0]=0 #set DC mode to zero
    k_arr[0][0][0]=0 #set k_arr back to zero for DC mode
    K_C1=np.multiply(Q_C1,J_C1)
    #In C, free Q_C1, J_C1
    if(equil!=0):
        kernel_equil+=-6*K_C1
    if(orthoCMB!=0):
        kernel_orthoCMB+=-24*K_C1
    #In C, free K_C1




    #Only one placement of the k weight is computed for K_B^I and K_B^II. Swapping G and H gives the same
    #real-space product (e.g. (d phi)*phi == phi*(d phi)), so the "second" kernels would equal the "first" ones,
    #and the coefficients below already refer to a single placement: 4*d^{-1}(phi d phi) + 2*nabla^{-2}(phi nabla^2 phi).
    #QuijotePNG paper's Table 4 writes K_B^I = (k_1+k_2)/k_3 (both placements, so i think using it with the coefficients 4 and 2
    #would unintetnionally double these terms?
    ##K_B^I (single placement: k_1/k_3)
    G_B1_first=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    H_B1_first=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    G_B1_first[:]=k_arr
    H_B1_first[:]=1
    Q_B1_first=kernel(G_B1_first,H_B1_first)

    J_B1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    k_arr[0][0][0]=1
    J_B1[:]=1/k_arr
    J_B1[0][0][0]=0
    k_arr[0][0][0]=0
    K_B1_first=np.multiply(Q_B1_first,J_B1)
    #In C, free Q_B1_first, J_B1
    if(equil!=0):
        kernel_equil+=4*K_B1_first
    if(orthoCMB!=0):
        kernel_orthoCMB+=10*K_B1_first
    #In C, free K_B1_first



    #Kernel K_B^II is also a sum of two different kernels, but they give the same result, so we just multiply these kernel's coefficients by 2
    ##K_B^II (single placement: k_1^2/k_3^2)
    G_B2_first=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    H_B2_first=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    G_B2_first[:]=np.multiply(k_arr,k_arr)
    H_B2_first[:]=1
    Q_B2_first=kernel(G_B2_first,H_B2_first)

    J_B2=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    k_arr[0][0][0]=1
    J_B2[:]=1/np.multiply(k_arr, k_arr)
    J_B2[0][0][0]=0
    k_arr[0][0][0]=0
    K_B2_first=np.multiply(Q_B2_first,J_B2)
    #In C, free Q_B2_first, J_B2
    if(equil!=0):
        kernel_equil+=2*K_B2_first
    if(orthoCMB!=0):
        kernel_orthoCMB+=8*K_B2_first
    #In C, free K_B2_first

    #calculating phi_{NG}(k) for f_NL^{equil}
    if(equil!=0):
        # kernel_equil = -3*K_A1 - 6*K_C1 + 4*K_B1 + 2*K_B2
        kernel_output = kernel_equil
        #free kernel_equil

    
    #calculating phi_{NG}(k) for f_NL^{orthoCMB}
    if(orthoCMB!=0):
        # kernel_orthoCMB = -9*K_A1 - 24*K_C1 + 10*K_B1 + 8*K_B2
        kernel_output = kernel_orthoCMB
        #free kernel_orthoCMB

##Derived assuming a scale invariant spectrum (n2=1)!!!
if(orthoLSS!=0):
    p=27/(-21 + (743)/(7*(20*np.pi**2-193)))
    t=(2+(60*p)/27)/(6*(1+(15*p)/27))

    #Kernel K_A^I
    G_A1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    H_A1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    G_A1[:]=1
    H_A1[:]=1
    Q_A1=kernel(G_A1, H_A1)
    J_A1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    J_A1[:]=1
    K_A1=np.multiply(Q_A1,J_A1)
    #In C, free Q_A1, J_A1
    kernel_orthoLSS=-(1+(9*p)/27)*K_A1
    #In C, free K_A1


    #Kernel K_C^I
    G_C1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    H_C1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    G_C1[:]=k_arr
    H_C1[:]=k_arr
    Q_C1=kernel(G_C1, H_C1)
    k_arr[0][0][0]=1 #momentarily modify the k_arr to avoid dividing by zero, which won't matter because we will set [0][0][0]=0 for the DC mode anyway
    J_C1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    J_C1=(1/3)*(1/np.multiply(k_arr,k_arr))
    J_C1[0][0][0]=0 #set DC mode to zero
    k_arr[0][0][0]=0 #set k_arr back to zero for DC mode
    K_C1=np.multiply(Q_C1,J_C1)
    #In C, free Q_C1, J_C1
    kernel_orthoLSS+=-(2+(60*p)/27)*K_C1
    #In C, free K_C1

    ##K_B^I (single placement: k_1/k_3)
    #Here, I'm assuming the similar x2 coefficient as in equil and ortho, but this would have to be checked
    G_B1_first=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    H_B1_first=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    G_B1_first[:]=k_arr
    H_B1_first[:]=1
    Q_B1_first=kernel(G_B1_first,H_B1_first)

    J_B1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    k_arr[0][0][0]=1
    J_B1[:]=1/k_arr
    J_B1[0][0][0]=0
    k_arr[0][0][0]=0
    K_B1_first=np.multiply(Q_B1_first,J_B1)
    #In C, free Q_B1_first, J_B1
    kernel_orthoLSS+=(1+(15*p)/27)*(1-t)*K_B1_first
    #In C, free K_B1_first


    #Kernel K_B^II is also a sum of two different kernels, but they give the same result, so we just multiply these kernel's coefficients by 2
    ##K_B^II (single placement: k_1^2/k_3^2)
    G_B2_first=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    H_B2_first=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    G_B2_first[:]=np.multiply(k_arr,k_arr)
    H_B2_first[:]=1
    Q_B2_first=kernel(G_B2_first,H_B2_first)

    J_B2=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    k_arr[0][0][0]=1
    J_B2[:]=1/np.multiply(k_arr, k_arr)
    J_B2[0][0][0]=0
    k_arr[0][0][0]=0
    K_B2_first=np.multiply(Q_B2_first,J_B2)
    #In C, free Q_B2_first, J_B2
    kernel_orthoLSS+=(1+(15*p)/27)*t*K_B2_first
    #In C, free K_B2_first

    #Kernel K_D^I
    G_D1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    H_D1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    k_arr[0][0][0]=1
    G_D1[:]=1/k_arr
    G_D1[0][0][0]=0
    k_arr[0][0][0]=0
    H_D1[:]=G_D1[:]
    Q_D1=kernel(G_D1, H_D1) 
    J_D1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    J_D1[:]=np.multiply(k_arr,k_arr)
    K_D1=np.multiply(Q_D1,J_D1)
    #In C, free Q_D1, J_D1
    kernel_orthoLSS+=(p/27)*K_D1
    #In C, free K_D1

    #Kernel K_E^II #sum of two identical kernels, assume the coefficeint is x2
    G_E2=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    H_E2=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    G_E2[:]=np.multiply(k_arr,k_arr)
    k_arr[0][0][0]=1
    H_E2[:]=1/k_arr
    H_E2[0][0][0]=0
    k_arr[0][0][0]=0
    Q_E2=kernel(G_E2, H_E2) 
    J_E2=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    k_arr[0][0][0]=1
    J_E2[:]=(1/2)*(1/k_arr)
    J_E2[0][0][0]=0
    k_arr[0][0][0]=0
    K_E2=np.multiply(Q_E2,J_E2)
    #In C, free Q_E2, J_E2
    kernel_orthoLSS+= -(20*p)/27*K_E2
    #In C, free K_E2

    #Kernel K_F^I,sum of two identical kernels, assuming coefficient is 2x
    G_F1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    H_F1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    k_arr[0][0][0]=1
    G_F1[:]=1/k_arr 
    G_F1[0][0][0]=0
    k_arr[0][0][0]=0
    H_F1[:]=1
    Q_F1=kernel(G_F1, H_F1) 
    J_F1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    J_F1[:]=k_arr
    K_F1=np.multiply(Q_F1,J_F1)
    #In C, free Q_F1, J_F1
    kernel_orthoLSS+=(6*p/27)*K_F1
    #In C, free K_F1

    #Kernel K_G^I,sum of two identical kernels, assuming coefficient is 2x
    G_G1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    H_G1=pyfftw.empty_aligned((N,N,N), dtype='complex128')
    k_arr[0][0][0]=1
    G_G1[:]=1/k_arr 
    G_G1[0][0][0]=0
    k_arr[0][0][0]=0
    H_G1[:]=k_arr
    Q_G1=kernel(G_G1, H_G1) 
    #J_G1 is 1, so we don't need to write it explicitly 
    K_G1=Q_G1
    #In C, free Q_G1
    kernel_orthoLSS+=(15*p/27)*K_G1
    #In C, free K_G1

    kernel_output=kernel_orthoLSS
    


os.makedirs(PNG_DIR, exist_ok=True)

##upload the kernel K (not Phi_NG) as y slices. The C load reads kernel_y.<y> from ZD_PNG_DIR.
# Each file is the (N, N) y-slice in [z][x] order (x fastest, then z), as interleaved float64 (real, imag),
# with no index columns.
for y in range(N//2 + 1):
    plane_zx = np.ascontiguousarray(kernel_output[:, y, :].T, dtype=file_dtype)   # [x, z] -> [z, x], C-contiguous
    filename = os.path.join(PNG_DIR, f"kernel_y.{y}")
    plane_zx.tofile(filename)
#free all remaining variables (phi_k_pad, k_arr, kernel_output, plane_zx)
print(f"Total run time: {time.perf_counter() - t_start:.2f} s")


