# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# defining required circuit parameters
R = 1 # resistance value
L = 100 # inductance value
T = 1 # time peiod of square wave (in seconds)
v_high = 10 # high voltage of square wave (in volts)
v_low = 0 # low voltage of square wave (in volts)
h = 0.0001 # step value (in seconds)

alphas = [0.5, 0.2, 0.8] # defining the duty ratios which we want to analyse
colors = ['b', 'r', 'g']  # assigning a colour for the response plot of each alpha value

# creating an array that defines the time-stamps (between 0 and 5 secs) at which we make observations
t = np.arange(0,5,h)
n = len(t) # total number of time-stamps

plt.figure(figsize=(8, 5)) # defining dimensions of the plot

# looping over different duty ratios to obtain response for each of them
for index, alpha in enumerate(alphas):
    # creating an array of voltage values at considered time-stamps
    V = np.where((t % T) < (alpha * T), v_high, v_low)

    # creating an array to store values of current that are calculated at the considered time-stamps
    i = np.zeros(n)

    # computing values of current i using trapezoidal method
    for j in range(n-1):
      i[j+1] = (i[j]*(1-(R*h)/(2*L)) + (h/(2*L))*(V[j] + V[j+1]))/(1 + (R*h)/(2*L))

    # plotting the response for each alpha (duty ratio)
    plt.plot(t, i, label=f"Duty Ratio = {alpha}", color=colors[index])

# formatting the plot
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8)  # Reference line at 0A
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
#plt.title("Transient Response of RL Circuit for $T<\\tau$")
plt.legend()
plt.grid()
plt.savefig('case3.pdf')
