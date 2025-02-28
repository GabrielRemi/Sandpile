import numpy as np
import matplotlib.pyplot as plt
from sandpile import *
from matplotlib.animation import FuncAnimation


system = SandpileND(dimension=2, linear_grid_size=20, critical_slope=7)

fig, ax = plt.subplots()
im = ax.imshow(np.zeros((20, 20)))


def update(frame):
    # system.step()
    # im.set_data(system._curr_slope)
    # im.set_data(np.random.randint(0, 100, size=(20, 20)))
    im.set_array(np.random.randint(0, 100, size=(20, 20)))
    return im


ani = FuncAnimation(fig, update, frames=100, interval=50)
# ani.save("test.mp4", writer="ffmpeg")

plt.show()
