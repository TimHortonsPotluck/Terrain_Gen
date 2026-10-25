import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import noise_funcs as nf

WIDTH = 500
HEIGHT = 500
main_scale = 5
amp_scale = 2
amp_offset_scale = .5
amp_offset_shift = 10

shelf_scale = 5
shelf_strength = .2

xs = np.linspace(0, 1, num=WIDTH, endpoint=False, dtype=np.float64)
ys = np.linspace(0, 1, num=HEIGHT, endpoint=False, dtype=np.float64)

X, Y = np.meshgrid(xs, ys)

amp_mults = nf.gradient_noise_np(amp_scale * X, amp_scale * Y)[0]
amp_mults = ((amp_mults - np.min(amp_mults)) / (np.max(amp_mults) - np.min(amp_mults))) # 0 to 1
amp_starts = amp_mults
amp_mults = .4 * amp_mults + .1

# amp_offset = np.cos(np.sqrt((X - .5) ** 2 + (Y - .5) ** 2) * np.pi * 2)
# amp_offset = np.where(np.sqrt((X - .5) ** 2 + (Y - .5) ** 2) <= .5, amp_offset, -np.ones_like(amp_offset))
# amp_offset = nf.gradient_noise_np(amp_offset_scale * X + amp_offset_shift, amp_offset_scale * Y + amp_offset_shift)[0]
# amp_offset = np.tanh(amp_offset)
amp_shelf = np.exp(-(9*((X - .5 + shelf_strength * nf.gradient_noise_np(shelf_scale * X + 5, shelf_scale * Y + 5)[0]) ** 2 + (Y - .5 + shelf_strength * nf.gradient_noise_np(shelf_scale * X + 10, shelf_scale * Y + 10)[0]) ** 2)) ** 2)

noise, dx, dy = nf.fractal_noise_warped_np(main_scale * X, main_scale * Y, warp_octaves=3, octaves=5, amp_start=amp_starts, amp_mult=.5)
# noise += amp_shelf
# noise += amp_starts
# noise, dx, dy = gradient_noise_np(main_scale * X, main_scale * Y)
# noise = rough_noise_np(main_scale * X, main_scale * Y)
# der = np.sqrt(dx ** 2 + dy ** 2)
# der_dir = np.atan2(dy, dx)
print("done making noise")
print("eroding")
nf.do_erode(noise, WIDTH / main_scale)
print("done eroding")
# plt.imshow(np.sqrt(dx ** 2 + dy ** 2))
# plt.imshow(np.atan2(dy, dx))
plt.imshow(noise)
# plt.imshow(amp_offset)
plt.show()

ndx, ndy = np.gradient(noise, WIDTH / main_scale)
ndtotal = np.sqrt(ndx ** 2 + ndy ** 2)

# fig = go.Figure(data=[
#     go.Surface(z=ndtotal, x=main_scale * X, y=main_scale * Y, colorscale='Viridis')
# ])
# fig.update_layout(scene=dict(aspectratio=dict(x=2, y=2, z=0.3)))
# fig.show()








