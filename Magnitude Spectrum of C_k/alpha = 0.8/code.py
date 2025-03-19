import numpy as np
import matplotlib.pyplot as plt

# Define alpha
alpha = 0.8

# Define range for k (excluding k=0 to avoid division by zero)
k_values = np.arange(-10, 10)

# Compute real and imaginary parts of C_k
Ck_real = (5 * np.sin(2 * np.pi * k_values * alpha)) / (np.pi * k_values)
Ck_imag = (5 / (np.pi * k_values)) * (np.cos(2 * np.pi * k_values * alpha) - 1)

# Compute magnitude of C_k
Ck_magnitude = np.sqrt(Ck_real**2 + Ck_imag**2)

# Plot magnitude spectrum
plt.figure(figsize=(8, 6))
plt.stem(k_values, Ck_magnitude, basefmt=" ")
plt.xlabel("Harmonic Index (k)")
plt.ylabel("Magnitude |C_k|")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig('spectrum3.png')
plt.show()
