# %%
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import scipy
from analysis import *
from analysis import draw_slope
from computation import auto_correlation, get_total_dissipation_rate
from sandpile import SandpileND
from utils import *

# %%
data_dir = pathlib.Path("data/d4_g15_c7_cl_nco")

# plt.ylim(10e-3)
# N = 2_000
sample_count = len(list(data_dir.glob("data_*.total_dissipation_rate.npy")))
print(sample_count)
f = np.load(data_dir / "data_0.total_dissipation_rate.npy")
n = len(f)

plt.figure(0, figsize=(12, 8))
# plt.plot(range(n), f)

# plt.subplot(223)
plt.xscale("log")
plt.yscale("log")
s = np.zeros(n // 2 + 1)

for i in range(sample_count):
    f = np.load(data_dir / f"data_{i}.total_dissipation_rate.npy")
    # corr = scipy.signal.correlate(f, f, mode="same", method="fft").astype(np.float64)
    # corr /= corr.max()
    # s += np.fft.rfft(corr).real
    s += np.fft.rfft(f).__abs__() ** 2
freq = np.fft.rfftfreq(n, 1)
plt.scatter(freq, s, s=2)

# plt.subplot(223)
# s /= samples
# plt.scatter(freq, s, s=1)
# # plt.plot(freq, s)

# # s = np.fft.fft(f).__abs__()**2
# # freq = np.fft.fftfreq(len(f), 1)
# # plt.scatter(freq, s, s=3)
# #
# m, b = calculate_scaling_exponent(freq, s, lower_limit=0.01, upper_limit=0.2)
# plt.plot(freq[1:], np.exp(b.nominal_value) * freq[1:] ** m.nominal_value)
# print(m)

# print(freq)

# %%
