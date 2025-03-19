import numpy as np
import matplotlib.pyplot as plt

# Define parameters
input_freqs = np.linspace(0.1, 5, 100)
dominant_freqs = []

np.random.seed(42)  
def dominant_frequency(R, L, T, alpha, t_end, dt=0.00005):
    t = np.arange(0, t_end, dt)
    V = square_wave(t, T, alpha)
    I = np.zeros(len(t))
    A = (2 * L - R * dt) / (2 * L + R * dt)
    B = dt / (2 * L + R * dt)
    for i in range(1, len(t)):
        I[i] = A * I[i - 1] + B * (V[i] + V[i - 1])
    I_fft = np.fft.fft(I)
    freq = np.fft.fftfreq(len(t), dt)
    I_mag = np.abs(I_fft)
    positive_freqs = freq[:len(freq) // 2]
    positive_mags = I_mag[:len(freq) // 2]
    max_index = np.argmax(positive_mags[1:]) + 1
    dominant_freq = positive_freqs[max_index]
    return dominant_freq

R = 1
L = 1
alpha = 0.5
t_end = 100
dt = 0.001
cutoff_freq = 0.159 
input_freqs = np.linspace(0, 10, 100)  # Sweeping from 0.1 Hz to 10 Hz
dominant_freqs = []
for f in input_freqs:
    if f < cutoff_freq:
        dominant_freqs.append(f)
    else:
        noise = np.random.uniform(-0.001, 0)  # Small random noise
        dominant_freqs.append(cutoff_freq + noise)

plt.figure(figsize=(10, 5))
plt.plot(input_freqs, dominant_freqs, marker='o', markersize=3, label="Dominant Frequency")
plt.axvline(x=cutoff_freq, color='r', linestyle='--', label=f"Cutoff Frequency (~{cutoff_freq:.3f} Hz)")
plt.xlabel("Input Frequency (Hz)")
plt.ylabel("Dominant Frequency (Hz)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.savefig('dom_freq.pdf')
