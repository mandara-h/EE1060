import numpy as np
import matplotlib.pyplot as plt

def square_wave(t, T, alpha):
    return np.array([10 if (time % T) < (alpha * T) else 0 for time in t])

def rl_circuit_response(R, L, T, alpha, t_end, dt=0.00005):
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
    dominant_mag = positive_mags[max_index]

    print(f"The frequency with the highest magnitude is approximately {dominant_freq:.2f} Hz")

    plt.figure(figsize=(10, 5))
    plt.plot(positive_freqs, positive_mags, label="Magnitude Spectrum")
    plt.scatter(dominant_freq, dominant_mag, color='red', label=f"Dominant Freq: {dominant_freq:.2f} Hz")
    plt.annotate(f"{dominant_freq:.3f} Hz",
                 (dominant_freq, dominant_mag),
                 xytext=(dominant_freq + 5, dominant_mag * 0.8),
                 arrowprops=dict(facecolor='red', arrowstyle="->"))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Fourier Transform of the Current")
    plt.xlim(0, 10)
    plt.grid(True)
    plt.legend()
    plt.savefig('fourier_coeffs.pdf')

R = 1
L = 1
T = 10
alpha = 0.5
t_end = 100
dt = 0.00005

rl_circuit_response(R, L, T, alpha, t_end, dt)
