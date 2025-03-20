import numpy as np
import matplotlib.pyplot as plt

# Function to compute the current response
def current_response(t, alpha, R, L, omega0, N=1000):
    a0 = 10 * alpha
    i_t = (a0 / R) * (1 - np.exp(-R * t / L))  # Initial response

    for n in range(1, N+1):
        theta = np.arctan(n * omega0 * L / R)
        an = (10 * np.sin(2 * np.pi * n * alpha)) / (np.pi * n)
        bn = (10 / (np.pi * n)) * (1 - np.cos(2 * np.pi * n * alpha))
        phi = np.arctan2(bn, an)
        magnitude = np.sqrt(an**2 + bn**2) / np.sqrt(R**2 + (n * omega0 * L)**2)

        # Summation term including sinusoidal components and exponential decay
        i_t += magnitude * (np.cos(n * omega0 * t - phi - theta) - np.exp(-R * t / L) * np.cos(phi + theta))

    return i_t

# Parameters
R = 1  # Resistance
L = 1  # Inductance
T = 1  # Time period
omega0 = 2 * np.pi / T  # Fundamental frequency

# Time values
t_values = np.linspace(0, 5, 1000)

# Different duty ratios to be plotted
alphas = [0.5, 0.2, 0.8]  
colors = ['blue', 'red', 'green']
labels = [r"Duty Ratio = 0.5", r"Duty Ratio = 0.2", r"Duty Ratio = 0.8"]

# Plot
plt.figure(figsize=(8, 6))

for alpha, color, label in zip(alphas, colors, labels):
    i_values = current_response(t_values, alpha, R, L, omega0)
    plt.plot(t_values, i_values, color=color, label=label)

# Formatting the plot
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.grid(True)
plt.axhline(0, color='black', linestyle='dashed', linewidth=1)  # Dashed horizontal line at y=0
plt.legend()

# Save and show the plot
plt.savefig('case1.pdf')
plt.show()
