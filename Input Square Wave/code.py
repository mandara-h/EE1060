import numpy as np
import matplotlib.pyplot as plt

def f(x, alpha, T, N=10000): #This function computes the Fourier series sum for f(x)
    result = 10 * alpha #The function starts with the DC term, which is 10alpha
    for n in range(1, N+1): #Loops from n=1 t0 N to approximate the infinite sum.
        term1 = (10 * np.sin(2 * np.pi * n * alpha) / (np.pi * n)) * np.cos((2 * np.pi * n / T) * x) #This represents the first term inside the summation
        term2 = (10 / (np.pi * n)) * (1 - np.cos(2 * np.pi * n * alpha)) * np.sin((2 * np.pi * n / T) * x)#This represents the second term inside the summation
        result += term1 + term2 #Updates result by adding the current term.
    return result

# Parameters
alpha = 0.5  # alpha = 0.5: A chosen value for alpha.Adjust as needed
T = 2 * np.pi #Defines the period of the function.
x_values = np.linspace(0, 5*T, 1000) #Generates 1000 points between -T and T for plotting

plt.figure(figsize=(8, 6))#Creates a new figure of size 8x6 inches.
y_values = f(x_values, alpha, T) #Calls f(x_values, alpha, T) to compute f(x)
plt.plot(x_values, y_values, color='blue')#plots f(x)

plt.title("Plot of f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.savefig('inputwave.pdf')
