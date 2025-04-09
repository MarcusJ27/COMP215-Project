import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Parameters
N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0
beta, gamma = 0.3, 0.1
t = np.linspace(0, 160, 160)

# Solve ODE
result = odeint(sir_model, [S0/N, I0/N, R0/N], t, args=(beta, gamma))
S, I, R = result.T

fig, ax = plt.subplots(figsize=(8, 5))
line_s, = ax.plot([], [], label='Susceptible')
line_i, = ax.plot([], [], label='Infected')
line_r, = ax.plot([], [], label='Recovered')
ax.set_xlim(0, t[-1])
ax.set_ylim(0, 1)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Proportion')
ax.set_title('Animated SIR Model Simulation')
ax.legend()
ax.grid(True)

def animate(frame):
    line_s.set_data(t[:frame], S[:frame])
    line_i.set_data(t[:frame], I[:frame])
    line_r.set_data(t[:frame], R[:frame])
    return line_s, line_i, line_r

ani = FuncAnimation(fig, animate, frames=len(t), interval=50, blit=True)

# Display in Colab
HTML(ani.to_jshtml())
