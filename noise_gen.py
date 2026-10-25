import matplotlib.pyplot as plt
import numpy as np
import scipy
import random

random.seed(0)

def value_noise(size_x, size_y, lat_size=32, num_levels=4, falloff=.5, interpolation="linear"): # size_x and size_y need to be a power of 2, and larger than lat_size, and 2^num_levels should be <= lat_size
    total_grid = [[0 for i in range(size_x)] for j in range(size_y)]
    mag = 1
    for l in range(num_levels + 1):
        level_lat_size = lat_size // 2 ** l
        num_level_lats_x = size_x // level_lat_size + 1 # the +1s are so the next lattice point can be present in the grid
        num_level_lats_y = size_y // level_lat_size + 1
        # print(num_level_lats_x, num_level_lats_y)
        level_lat_grid = [[random.random() * 2 - 1 for i in range(num_level_lats_x)] for j in range(num_level_lats_y)] 
        level_grid = [[0 for i in range(size_x)] for j in range(size_y)]
        # print(level_lat_grid)
        # interpolate
        for cell_x in range(num_level_lats_x - 1):
            for cell_y in range(num_level_lats_y - 1):
                # print(cell_x, cell_y)
                f11 = level_lat_grid[cell_y][cell_x]
                f12 = level_lat_grid[(cell_y + 1)][cell_x]
                f21 = level_lat_grid[cell_y][(cell_x + 1)]
                f22 = level_lat_grid[(cell_y + 1)][(cell_x + 1)]
                for i in range(level_lat_size):
                    for j in range(level_lat_size):
                        if i == 0 and j == 0: # this is a lattice point
                            level_grid[j + cell_y * level_lat_size][i + cell_x * level_lat_size] = f11
                        x2 = level_lat_size - 1
                        y2 = level_lat_size - 1
                        level_grid[j + cell_y * level_lat_size][i + cell_x * level_lat_size] = (f11 * (x2 - i) * (y2 - j) + f12 * (x2 - i) * j + f21 * i * (y2 - j) + f22 * i * j) / level_lat_size ** 2
        for i in range(size_x):
            for j in range(size_y):
                total_grid[j][i] += level_grid[j][i] * mag
        mag *= falloff
    
    print(max(map(max, total_grid)))
    print(min(map(min, total_grid)))
    for i in range(size_x):
        for j in range(size_y):
            total_grid[j][i] = min(1, max(-1, total_grid[j][i]))
    return total_grid

def random_noise(size_x, size_y):
    return [[random.random() for i in range(size_x)] for j in range(size_y)]

grid = value_noise(256, 256, lat_size=32, num_levels=4)
rand_grid = random_noise(32, 32)

rect_in_grid = [[0 for i in range(32)] for j in range(32)]
for i in range(2):
    for j in range(4):
        rect_in_grid[j + 14][i + 15] = 1

def remove_high_freqs_real(rfft2, r, type="square"): # must be centered, i.e. have had fftshift applied
    
    size_x = len(rfft2[0])
    size_y = len(rfft2)
    new_rfft2 = [[0 for i in range(size_x)] for j in range(size_y)]
    half_size_x = size_x // 2
    half_size_y = size_y // 2
    adj_x = size_x + 1
    adj_y = size_y + 1
    if type == "square":
        for x in range(r):
            for y in range(-r, r):
                new_rfft2[y + half_size_y][x] = rfft2[y + half_size_y][x]
    elif type == "circle":
        for x in range(r+1):
            for y in range(-r, r+1):
                i = x
                j = y + half_size_y
                if (x) ** 2 + (y) ** 2 <= (r) ** 2 and i < size_x and j >= 0 and j < size_y:
                    new_rfft2[j][i] = rfft2[j][i]
    return new_rfft2
    
plt.imshow(grid, vmin=-1, vmax=1)
plt.show()
rfft2_grid = scipy.fft.rfft2(grid)
rfft2_grid = scipy.fft.fftshift(rfft2_grid, axes=(0))
plt.imshow(np.abs(rfft2_grid))
plt.show()

# filtered_rfft2_grid = remove_high_freqs_real(rfft2_grid, 16, type="circle")
# plt.imshow(np.abs(filtered_rfft2_grid))
# plt.show()

# recon_grid = scipy.fft.irfft2(scipy.fft.ifftshift(filtered_rfft2_grid, axes=(0)))
# rfft2_grid_real = np.real(rfft2_grid)
# rfft2_grid_imag = np.imag(rfft2_grid)
# rfft2_grid_mag = np.abs(rfft2_grid)
# rfft2_grid_phase = np.atan2(rfft2_grid_imag, rfft2_grid_real)

# adj_rfft2_grid_phase = rfft2_grid_phase
# adj_rfft2_grid_real = rfft2_grid_mag * np.cos(adj_rfft2_grid_phase)
# adj_rfft2_grid_imag = rfft2_grid_mag * np.sin(adj_rfft2_grid_phase)

# recon_rfft2_grid_mag = adj_rfft2_grid_real + 1j * adj_rfft2_grid_imag

# recon_grid = scipy.fft.irfft2(scipy.fft.ifftshift(recon_rfft2_grid_mag, axes=(0)))

# plt.imshow(recon_grid, vmin=-1, vmax=1)
# plt.show()

# fft2_grid = scipy.fft.fft2(grid)
# fft2_grid = scipy.fft.fftshift(fft2_grid)
# fft2_grid_real = np.real(fft2_grid)
# fft2_grid_imag = np.imag(fft2_grid)
# fft2_grid_mag = np.abs(fft2_grid)
# fft2_grid_phase = np.atan2(fft2_grid_imag, fft2_grid_real)
# plt.imshow(grid, vmin=-1, vmax=1)
# # plt.imshow(fft2_grid_real)
# # plt.imshow(fft2_grid_real)
# # plt.imshow(fft2_grid_imag)
# # plt.imshow(fft2_grid_mag)
# # plt.imshow(fft2_grid_phase)
# plt.show()
# recon_fft2 = remove_high_freqs(fft2_grid, 32)
# # plt.imshow(np.real(recon_fft2))
# # plt.show()
# # print(recon_fft2)
# # scipy.signal.convolve2d(fft2_grid_mag, scharr, boundary='fill', mode='same')
# # adj_fft2_grid_mag = scipy.ndimage.gaussian_filter(fft2_grid_mag, sigma=1, mode='constant')
# # adj_fft2_grid_phase = scipy.ndimage.gaussian_filter(fft2_grid_phase, sigma=1, mode='constant')
# # recon_fft2_real = adj_fft2_grid_mag * np.cos(fft2_grid_phase)
# # recon_fft2_imag = adj_fft2_grid_mag * np.sin(fft2_grid_phase)
# # recon_fft2 = scipy.fft.ifftshift(recon_fft2_real + 1j * recon_fft2_imag)
# recon_grid = np.real(scipy.fft.ifft2(recon_fft2))
# plt.imshow(recon_grid, vmin=-1, vmax=1)
# plt.show()
# # scipy.signal.convolve2d()

