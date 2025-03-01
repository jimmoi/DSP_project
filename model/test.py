import matplotlib.pyplot as plt
import numpy as np
import time

# Signal parameters
t = 10    # Duration in seconds
sr = 20   # Sampling rate
time_vals = np.linspace(0, t, t * sr)  # Time values

f = 1  # Frequency in Hz
a = 3  # Amplitude
signal = a * np.sin(2 * np.pi * f * time_vals)  # Sine wave signal

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, t)
ax.set_ylim(-a - 1, a + 1)

# Create the scatter plot
point, = ax.plot([], [], 'ro')  # 'ro' means red circle

# Animation loop
for i in range(len(time_vals)):
    point.set_data([time_vals[i]], [signal[i]])  # Update point position
    plt.pause(1/60)  # Small delay to create animation effect

plt.show()
