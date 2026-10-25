import numpy as np

rng = np.random.default_rng(seed=0)
lookup_len = 1024
lookup_np = np.array(range(lookup_len))
rng.shuffle(lookup_np)

def lin_interp(a, b, t): # assume dist 1
    return a + (b - a) * t

def bilin_interp(a, b, c, d, tx, ty):
    return lin_interp(lin_interp(a, b, tx), lin_interp(c, d, tx), ty)

def linear_shaping(t):
    return t

def cubic_shaping(t):
    return (3 - 2 * t) * t ** 2

def quintic_shaping(t):
    return ((6 * t - 15 ) * t + 10) * t ** 3
    
def quintic_shaping_der(t):
    return ((t - 2) * t + 1) * 30 * t ** 2

def skew(x, k=2.0):
    return np.sign(x) * (np.abs(x) ** k)

def exp_skew(x, strength=2.0):
    return np.tanh(x * strength)

def exp_skew2(x, strength=2.0):
    return np.exp(x * strength) - 1.0

def hsh_1d_np(i):
    i2 = np.mod(i, lookup_len)
    return lookup_np[i2]

def hsh_2d_np(i, j):
    i2 = np.mod(i, lookup_len)
    j2 = np.mod(j, lookup_len)
    return lookup_np[i2 ^ lookup_np[j2]]

def hsh_vec_np(i, j): # random-looking vector
    angle = 2 * np.pi * (hsh_2d_np(i, j) / lookup_len)
    return np.stack((np.cos(angle), np.sin(angle)), axis=-1)

def rough_noise_np(x, y): # x anad y are the coordinates from a numpy array/meshgrid
    x = np.asarray(x)
    y = np.asarray(y)
    i = x.astype(int)
    j = y.astype(int)
    return (hsh_2d_np(i, j) / (lookup_len - 1)) # -1 to 1

def gradient_noise_np(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    i = np.floor(x).astype(int)
    j = np.floor(y).astype(int)
    i_frac = x - i
    j_frac = y - j
    g00 = hsh_vec_np(i, j)
    g10 = hsh_vec_np(i + 1, j)
    g01 = hsh_vec_np(i, j + 1)
    g11 = hsh_vec_np(i + 1, j + 1)

    f00 = g00[..., 0] * i_frac + g00[...,1] * j_frac
    f10 = g10[..., 0] * (i_frac - 1) + g10[...,1] * j_frac
    f01 = g01[..., 0] * i_frac + g01[...,1] * (j_frac - 1)
    f11 = g11[..., 0] * (i_frac - 1) + g11[...,1] * (j_frac - 1)
    
    tx = quintic_shaping(i_frac)
    ty = quintic_shaping(j_frac)
    
    val = bilin_interp(f00, f10, f01, f11, tx, ty)
    
    gx = bilin_interp(g00[..., 0], g10[..., 0], g01[..., 0], g11[..., 0], tx, ty)
    gy = bilin_interp(g00[..., 1], g10[..., 1], g01[..., 1], g11[..., 1], tx, ty)
    
    bilin_dx = lin_interp(f10 - f00, f11 - f01, ty)
    bilin_dy = lin_interp(f01 - f00, f11 - f10, tx)
    
    dx = gx + bilin_dx * quintic_shaping_der(i_frac)
    dy = gy + bilin_dy * quintic_shaping_der(j_frac)
    
    return val, dx, dy

def fractal_noise_np(x, y, octaves=5, amp_start=1., amp_mult=.5, freq_mult=2):
    x = np.asarray(x)
    y = np.asarray(y)
    amp_start = np.asarray(amp_start)
    amp_mult = np.asarray(amp_mult)
    freq_mult = np.asarray(freq_mult)
    
    val_sums = np.zeros_like(x)
    dx_sums = np.zeros_like(x)
    dy_sums = np.zeros_like(x)
    amps = np.ones_like(amp_mult) * amp_start
    freqs = np.ones_like(freq_mult)
    
    for i in range(octaves):
        vals, dxs, dys = gradient_noise_np(x * freqs, y * freqs)
        val_sums += amps * vals
        dx_sums += amps * dxs * freqs
        dy_sums += amps * dys * freqs
        amps *= amp_mult
        freqs *= freq_mult
    return val_sums, dx_sums, dy_sums

def fractal_noise_warped_np(x, y, warp_scale=1, warp_strength=.1, warp_octaves=3, warp_freq_mult=2, warp_amp_mult=.5, octaves=5, amp_start=1., amp_mult=.5, freq_mult=2):
    x = np.asarray(x)
    y = np.asarray(y)
    amp_start = np.asarray(amp_start)
    amp_mult = np.asarray(amp_mult)
    freq_mult = np.asarray(freq_mult)
    
    val_sums = np.zeros_like(x)
    dx_sums = np.zeros_like(x)
    dy_sums = np.zeros_like(x)
    amps = np.ones_like(amp_mult)
    freqs = np.ones_like(freq_mult)
    
    warp_x = np.zeros_like(x)
    warp_y = np.zeros_like(x)
    for i in range(warp_octaves):
        warp_x_vals, warp_x_dxs, warp_x_dys = gradient_noise_np(x * freqs / warp_scale + 10, y * freqs / warp_scale + 10) # 10 to change the "seed"
        warp_y_vals, warp_y_dxs, warp_y_dys = gradient_noise_np(x * freqs / warp_scale + 20, y * freqs / warp_scale + 20) # 20 to change the "seed"
        warp_x += amps * warp_x_vals
        warp_y += amps * warp_y_vals
        amps *= warp_amp_mult
        freqs *= warp_freq_mult
    amps = np.ones_like(amp_mult) * amp_start # we can reuse these for the main terrain noise
    freqs = np.ones_like(freq_mult) # we can reuse these for the main terrain noise
    for i in range(octaves):
        warped_x = warp_x * warp_strength + x
        warped_y = warp_y * warp_strength + y
        vals, dxs, dys = gradient_noise_np(warped_x * freqs, warped_y * freqs)
        val_sums += amps * vals
        dx_sums += amps * dxs * freqs
        dy_sums += amps * dys * freqs
        amps *= amp_mult
        freqs *= freq_mult
    return val_sums, dx_sums, dy_sums

def do_erode(arr, h, iterations=5):
    # also doing this as not vectorized rn
    
    # plan: 
    #   computge gradient for each iteration. only update terrain heights at end of each iteration
    #   each cell gets a small amount of water per iteration
    #   the water flows downhill, picking up or depositing sediment based on slope
    #   drops that run into each other combine
    #   evaporate some water away
    
    # some params:
    drop_volume = .5 
    drop_capacity = 1 # fraction of volume
    deposit_rate = .05
    erosion_rate = .5
    evap_rate = .1 # per time step
    water = np.zeros_like(arr) * drop_volume
    next_water = np.zeros_like(arr)
    sediment = np.zeros_like(arr)
    water[350][125] = drop_volume # testing
    for i in range(iterations):
        print(f"{i} / {iterations}")
        dx, dy = np.gradient(arr, h)
        dtot = np.sqrt(dx ** 2 + dy ** 2)
        dxnorm = dx / dtot
        dynorm = dy / dtot
        for x in range(water.shape[0]):
            for y in range(water.shape[1]):
                local_water = water[y][x]
                flow_dir_x = -dxnorm[y][x]
                flow_dir_y = -dynorm[y][x]
                slope = dtot[y][x]
                erode_amt = dtot[y][x] * erosion_rate
                deposit_amt = sediment[y][x] * deposit_rate
                
        
