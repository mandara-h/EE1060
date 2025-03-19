import numpy as np
import matplotlib.pyplot as plt

def current_response(t, alpha, R, L, omega0, N=10000):
    a0 = 10 * alpha
    i_t = 0

    for n in range(1, N+1):
        theta = np.arctan(n * omega0 * L / R)
        an = (10 * np.sin(2 * np.pi * n * alpha)) / (np.pi * n)
        bn = (10 / (np.pi * n)) * (1 - np.cos(2 * np.pi * n * alpha))
        phi = np.arctan2(bn , an)
        magnitude = np.sqrt(an**2 + bn**2) / np.sqrt(R**2 + (n * omega0 * L)**2)
        phase = n * omega0 * t - phi - theta
        decay_term = np.exp(-R * t / L) * np.cos(phi + theta) / np.sqrt(R**2 + (n * omega0 * L)**2)
        i_t += magnitude * np.cos(phase) - decay_term

    i_t += (a0/R)*(1 - np.e**(-R * t / L))

    return i_t

# Parameters
alpha = 0.5
R = 1  # Resistance
L = 1  # Inductance
T = 1  # Time period
omega0 = 2 * np.pi / T  # Fundamental frequency (Now ω₀ = 2π)

t_values = np.linspace(0, 5, 1000)
i_values = current_response(t_values, alpha, R, L, omega0)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(t_values, i_values, color='blue')
plt.xlabel("Time (t)")
plt.ylabel("Current i(t)")
plt.grid(True)
plt.savefig('analytical_sol.pdf')
