import matplotlib.pyplot as plt
import numpy as np
import scipy
import random
import tkinter as tk
from tkinter import ttk
import plotly.graph_objects as go

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

# ===== non-vectorized

def hsh_1d(i):
    i2 = i % lookup_len
    return lookup[i2]

def hsh_2d(i, j):
    i2 = i % lookup_len
    j2 = j % lookup_len
    return hsh_1d(i2 ^ hsh_1d(j2))

def hsh_vec(i, j): # random-looking vector
    angle = 2 * np.pi * (hsh_2d(i, j) / lookup_len)
    return (np.cos(angle), np.sin(angle))

def dot(v0, v1): # v0 and v1 must be 2 element tuples
    return v0[0] * v1[0] + v0[1] * v1[1]

def rough_noise(x, y):
    i = int(x)
    j = int(y)
    return (hsh_2d(i, j) / (lookup_len - 1)) # -1 to 1

def value_noise(x, y, shaping):
    i = int(x)
    j = int(y)
    i_frac = x - i
    j_frac = y - j
    f00 = hsh_2d(i, j) / (lookup_len - 1)
    f10 = hsh_2d(i + 1, j) / (lookup_len - 1)
    f01 = hsh_2d(i, j + 1) / (lookup_len - 1)
    f11 = hsh_2d(i + 1, j + 1) / (lookup_len - 1)
    return bilin_interp(f00, f10, f01, f11, shaping(i_frac), shaping(j_frac))

def gradient_noise(x, y):
    i = int(x)
    j = int(y)
    i_frac = x - i
    j_frac = y - j
    g00 = hsh_vec(i, j)
    g10 = hsh_vec(i + 1, j)
    g01 = hsh_vec(i, j + 1)
    g11 = hsh_vec(i + 1, j + 1)
    f00 = dot(g00, (i_frac, j_frac))
    f10 = dot(g10, (i_frac - 1, j_frac))
    f01 = dot(g01, (i_frac, j_frac - 1))
    f11 = dot(g11, (i_frac - 1, j_frac - 1))
    
    tx = quintic_shaping(i_frac)
    ty = quintic_shaping(j_frac)
    
    val = bilin_interp(f00, f10, f01, f11, tx, ty)
    
    gx = bilin_interp(g00[0], g10[0], g01[0], g11[0], tx, ty)
    gy = bilin_interp(g00[1], g10[1], g01[1], g11[1], tx, ty)
    
    bilin_dx = lin_interp(f10 - f00, f11 - f01, ty)
    bilin_dy = lin_interp(f01 - f00, f11 - f10, tx)
    
    dx = gx + bilin_dx * quintic_shaping_der(i_frac)
    dy = gy + bilin_dy * quintic_shaping_der(j_frac)
    
    return val, dx, dy

def fractal_noise(x, y, octaves=5, amp_mult=.5, freq_mult=2):
    sum = 0
    amp = 1
    freq = 1
    for i in range(octaves):
        sum += amp * gradient_noise(x * freq, y * freq)[0]
        amp *= amp_mult
        freq *= freq_mult
    return sum

def fractal_noise_grad(x, y, octaves=5, amp_mult=.5, freq_mult=2, grad="x"):
    sum = 0
    amp = 1
    freq = 1
    if grad == "total":
        sumx = 0
        sumy = 0
        for i in range(octaves):
            vals = gradient_noise(x * freq, y * freq) # val, dx, dy
            sumx += amp * vals[1] * freq
            sumy += amp * vals[2] * freq
            amp *= amp_mult
            freq *= freq_mult
        return np.sqrt(sumx ** 2 + sumy ** 2)
    else:
        if grad == "x":
            d = 1
        if grad == "y":
            d = 2
        for i in range(octaves):
            sum += amp * gradient_noise(x * freq, y * freq)[d] * freq
            amp *= amp_mult
            freq *= freq_mult
        return sum

def fake_erosion(x, y, octaves=5, amp_mult=.5, freq_mult=2):
    sum = 0
    amp = 1
    freq = 1
    dx_sum = 0
    dy_sum = 0
    for i in range(octaves):
        vals = gradient_noise(x * freq, y * freq) # val, dx, dy
        dx_sum += vals[1]
        dy_sum += vals[2]
        weight = 1 / (1 + dx_sum ** 2 + dy_sum ** 2)
        sum += amp * vals[0] * weight
        amp *= amp_mult
        freq *= freq_mult
    return sum

# ===== vectorized

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

def fractal_noise_warped_np(x, y, warp_scale=1, warp_strength=.1, warp_octaves=3, octaves=5, amp_start=1., amp_mult=.5, freq_mult=2):
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
    
    warp_freq_mult = 2
    warp_amp_mult = .5
    
    warp_x = np.zeros_like(x)
    warp_y = np.zeros_like(x)
    for i in range(warp_octaves):
        warp_x += amps * gradient_noise_np(x * freqs / warp_scale + 10, y * freqs / warp_scale + 10)[0] # 10 to change the "seed"
        warp_y += amps * gradient_noise_np(x * freqs / warp_scale + 20, y * freqs / warp_scale + 20)[0] # 20 to change the "seed"
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

def fractal_noise_terrain_np(x, y, octaves=5, amp_start=1., amp_mult=.5, freq_mult=2, erosion=0): # not quite working
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
        # vals = exp_skew2(vals)
        weights = np.ones_like(x) / (1 + (dx_sums ** 2 + dy_sums ** 2) * erosion)
        val_sums += amps * vals * weights
        dx_sums += amps * dxs * freqs
        dy_sums += amps * dys * freqs
        amps *= amp_mult
        freqs *= freq_mult
    return val_sums, dx_sums, dy_sums


##### DONT USE THIS ONE FOR NOW
##### USE THE FUNCTIONS IN NOISE_FUNCS

WIDTH = 1000
HEIGHT = 1000
main_scale = 5
amp_scale = 2
amp_offset_scale = .5
amp_offset_shift = 10

highlands_scale = 1

xs = np.linspace(0, 1, num=WIDTH, endpoint=False, dtype=np.float64)
ys = np.linspace(0, 1, num=HEIGHT, endpoint=False, dtype=np.float64)

X, Y = np.meshgrid(xs, ys)

amp_mults = gradient_noise_np(amp_scale * X, amp_scale * Y)[0]
amp_mults = ((amp_mults - np.min(amp_mults)) / (np.max(amp_mults) - np.min(amp_mults))) # 0 to 1
amp_starts = amp_mults
amp_mults = .4 * amp_mults + .1

amp_offset = gradient_noise_np(amp_offset_scale * X + amp_offset_shift, amp_offset_scale * Y + amp_offset_shift)[0]




noise, dx, dy = fractal_noise_warped_np(main_scale * X, main_scale * Y, octaves=5, amp_start=amp_starts, amp_mult=.5)
# noise = np.sqrt(noise ** 2 + 1) - 1
# noise += np.abs(amp_offset)
# noise, dx, dy = gradient_noise_np(main_scale * X, main_scale * Y)
# noise = rough_noise_np(main_scale * X, main_scale * Y)
# der = np.sqrt(dx ** 2 + dy ** 2)
# der_dir = np.atan2(dy, dx)
print("done making noise")

# plt.imshow(np.sqrt(dx ** 2 + dy ** 2))
# plt.imshow(np.atan2(dy, dx))
# plt.imshow(noise)
# plt.imshow(amp_offset)
# plt.show()


fig = go.Figure(data=[
    go.Surface(z=noise, x=main_scale * X, y=main_scale * Y, colorscale='Viridis')
])
fig.update_layout(scene=dict(aspectratio=dict(x=2, y=2, z=0.3)))
fig.show()


""" not vectorized
# grid = [[0 for i in range(WIDTH)] for j in range(HEIGHT)]

# for i in range(WIDTH):
#     for j in range(HEIGHT):
#         # val = fractal_noise_grad(i / scale, j / scale, octaves=5, amp_mult=.51, freq_mult=1.98, grad="total")
#         val = fake_erosion(i / scale, j / scale, octaves=5, amp_mult=.51, freq_mult=1.98)
#         # val = fractal_noise(i / scale, j / scale, octaves=5, amp_mult=.51, freq_mult=1.98)
#         grid[j][i] = val

# print(max(map(max, grid)))
# print(min(map(min, grid)))
# plt.imshow(grid)
# plt.show()

"""