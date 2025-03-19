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
    plt.figure(figsize=(10, 5))
    plt.plot(t, I, label="Current (I)")
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A)")
    plt.grid(True)
    plt.legend()
    plt.savefig('numerical_sol.pdf')

R = 1
L = 1
T = 1
alpha = 0.5
t_end = 5
dt = 0.00005

rl_circuit_response(R, L, T, alpha, t_end, dt)
