import numpy as np
import matplotlib.pyplot as plt

def square_wave(t, T, alpha):
    """Generates a square wave with period T and duty cycle alpha."""
    return np.array([10 if (time % T) < (alpha * T) else 0 for time in t])

def dominant_frequency(R, L, T, alpha, t_end, dt):
    """Calculates the dominant frequency of the RL circuit's response."""
    t = np.arange(0, t_end, dt)
    V = square_wave(t, T, alpha)
    I = np.zeros(len(t))

    # Coefficients for RL circuit update
    A = (2 * L - R * dt) / (2 * L + R * dt)
    B = dt / (2 * L + R * dt)

    # Simulate current response over time
    for i in range(1, len(t)):
        I[i] = A * I[i - 1] + B * (V[i] + V[i - 1])

    # Perform FFT to analyze the frequency content
    n_samples = max(2**14, len(t))  # Increase FFT resolution with zero padding
    I_fft = np.fft.fft(I, n=n_samples)
    freq = np.fft.fftfreq(n_samples, dt)
    I_mag = np.abs(I_fft)

    # Consider only positive frequencies
    positive_freqs = freq[:len(freq) // 2]
    positive_mags = I_mag[:len(freq) // 2]

    # Ignore DC component (0 Hz) and get the dominant frequency
    max_index = np.argmax(positive_mags[1:]) + 1
    dominant_freq = positive_freqs[max_index]

    # Ignore low or near-zero frequencies or small magnitudes
    if dominant_freq < 0.1 or np.max(positive_mags[1:]) < 1e-3:  
        dominant_freq = 0

    return dominant_freq

# Circuit parameters
R = 1
L = 1
alpha = 0.5

# Sweep input frequencies
input_freqs = np.linspace(0.1, 100, 50)  # Sweeping from 0.1 Hz to 100 Hz
dominant_freqs = []

# Loop over input frequencies and calculate dominant frequencies
for f in input_freqs:
    T = 1 / f
    dt = min(1 / (500 * f), 0.001)  # Fine time step to avoid aliasing
    t_end = max(100, 100 * T)  # Simulate for at least 100 periods
    dominant_freqs.append(dominant_frequency(R, L, T, alpha, t_end, dt))

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(input_freqs, dominant_freqs, marker='o', label="Dominant Frequency")
plt.plot(input_freqs, input_freqs, linestyle='--', color='red', label="Input Frequency (Ideal)")
plt.xlabel("Input Frequency (Hz)")
plt.ylabel("Dominant Frequency (Hz)")
plt.grid(True)
plt.legend()
plt.savefig('dom_freq.png')
plt.show()
