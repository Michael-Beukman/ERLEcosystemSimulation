import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x = np.arange(130, 190, 1)
y = 97.928 * np.exp(- np.exp(-  0.1416 *( x - 146.1 )))
z = 96.9684 * np.exp(- np.exp(-0.1530*( x - 144.4)))

fig, ax = plt.subplots()
line1, = ax.plot(x, y, color = "r")
# line2, = ax.plot(x, z, color = "g")

def update(num, x, y, z):
    line1.set_data(x[:num], y[:num])
    # line2.set_data(x[:num], z[:num])
    return [line1]

ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, z],
                  interval=295, blit=True)

ax.set_xlabel('Age (day)')
ax.set_ylabel('EO (%)')

plt.show()