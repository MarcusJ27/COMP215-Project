# ---------------------------------- imports ---------------------------------- 
import numpy as np
import sympy as sp
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# ---------------------------------- Setup for calcs ---------------------------------- 
# 2d Vector class
class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return f"({self.x}, {self.y})"
    def __add__(self, v):
        return Vec2(self.x + v.x, self.y + v.y)
    def __radd__(self, v):
        return Vec2(self.x + v.x, self.y + v.y)
    def __sub__(self, v):
        return Vec2(self.x - v.x, self.y - v.y)
    def __rsub__(self, v):
        return Vec2(v.x - self.x, v.y - self.y)
    def __mul__(self, n):
        return Vec2(self.x * n, self.y * n)
    def __rmul__(self, n):
        return Vec2(self.x * n, self.y * n)
    def dot(self, v):
        return self.x * v.x + self.y * v.y
    def get_length(self):
        return np.sqrt(self.dot(self))

# Particle class to represent each planet / setup placeholders for acel, solutions, and lambda
class Particle():
    n = 0
    def __init__(self, initial_pos, initial_vel, mass):
        self.i = Particle.n
        Particle.n += 1

        self.m = mass
        self.G = 1

        self.pos = Vec2(sp.symbols("x_" + str(self.i)), sp.symbols("y_" + str(self.i)))
        self.vel = Vec2(sp.symbols("vx_" + str(self.i)), sp.symbols("vy_" + str(self.i)))
        self.acc = Vec2(0, 0)

        self.lamb_vel = Vec2(None, None)
        self.lamd_acc = Vec2(None, None)

        self.initial_pos = initial_pos
        self.initial_vel = initial_vel

        self.vf_vel = Vec2(0, 0)
        self.vf_acc = Vec2(0, 0)

        self.sol_pos = Vec2(None, None)
        self.sol_vel = Vec2(None, None)

    def calculate_acc(self, particles):
        for j in range(len(particles)):
            if self.i != j:
                self.acc += (particles[j].pos - self.pos) * particles[j].m * self.G * (
                    1 / (((self.pos.x - particles[j].pos.x) ** 2 + (self.pos.y - particles[j].pos.y) ** 2) ** (3 / 2)))

    def lambdify_vel(self, particles):
        self.lamb_vel.x = sp.lambdify(self.vel.x, self.vel.x)
        self.lamb_vel.y = sp.lambdify(self.vel.y, self.vel.y)

    def lambdify_acc(self, particles):
        var = []
        for j in range(len(particles)):
            var.append(particles[j].pos.x)
            var.append(particles[j].pos.y)

        self.lamd_acc.x = sp.lambdify([var], self.acc.x)
        self.lamd_acc.y = sp.lambdify([var], self.acc.y)

# ---------------------------------- User Input ----------------------------------


num_planets = int(input("Number of planets: "))

par = []
Particle.n = 0

for _ in range(num_planets):
    pos = Vec2(random.uniform(5, 15), random.uniform(5, 15))
    vel = Vec2(random.uniform(0.1, 1), random.uniform(0.1, 1))
    mass = random.uniform(0.5, 1)
    par.append(Particle(initial_pos=pos, initial_vel=vel, mass=mass))

t_end = 60.0
steps = 1000

# ---------------------------------- Calculations  ---------------------------------- 

n = len(par)

# Symbolic Calculation(converts accel and vel to lamda)
for i in range(n):
    par[i].calculate_acc(par)
    par[i].lambdify_vel(par)
    par[i].lambdify_acc(par)

# vectorfield for ODE
def vectorfield(var, t):
    pos = var[0:2 * n]
    vel = var[2 * n:4 * n]
    f = []

    for i in range(n):
        par[i].vf_vel.x = par[i].lamb_vel.x(vel[2 * i])
        par[i].vf_vel.y = par[i].lamb_vel.y(vel[2 * i + 1])
        f.append(par[i].vf_vel.x)
        f.append(par[i].vf_vel.y)

    for i in range(n):
        par[i].vf_acc.x = par[i].lamd_acc.x(pos)
        par[i].vf_acc.y = par[i].lamd_acc.y(pos)
        f.append(par[i].vf_acc.x)
        f.append(par[i].vf_acc.y)

    return f

# Inital conditions
var = []
for i in range(n):
    var.append(par[i].initial_pos.x)
    var.append(par[i].initial_pos.y)
for i in range(n):
    var.append(par[i].initial_vel.x)
    var.append(par[i].initial_vel.y)

# ODE solver parameters
t = np.linspace(0, t_end, steps + 1)
sol = odeint(vectorfield, var, t)
sol = np.transpose(sol)


for i in range(n):
    par[i].sol_pos.x = sol[2 * i]
    par[i].sol_pos.y = sol[2 * i + 1]
    par[i].sol_vel.x = sol[2 * n + 2 * i]
    par[i].sol_vel.y = sol[2 * n + 2 * i + 1]

# Energy calc for sim/ saved in array to verify conversion 
Energy = 0 
for i in range(0, n):
    for j in range(i+1, n):
        Energy += (-1/(((par[i].sol_pos.x-par[j].sol_pos.x)**2 + (par[i].sol_pos.y-par[j].sol_pos.y)**2)**(1/2)))

for i in range(0, n):
    Energy += 0.5*(par[i].sol_vel.x*par[i].sol_vel.x + par[i].sol_vel.y*par[i].sol_vel.y)

# ---------------------------------- Visualization ---------------------------------- 

plt.style.use('dark_background')
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

ax.axis('equal')
ax.axis([0, 40, 0, 40])
ax.set_title(f'N-body Gravitational Simulation | Number of Planets: {n}', pad=20)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)


circle = [None] * n
line = [None] * n
colors = plt.cm.hsv(np.linspace(0, 1, n))  

for i in range(n):
    circle[i] = plt.Circle((par[i].sol_pos.x[0], par[i].sol_pos.y[0]), 0.08,
                           ec="w", lw=2.5, zorder=20, fc=colors[i])
    ax.add_patch(circle[i])
    line[i] = ax.plot([], [], color=colors[i], alpha=0.7, lw=1)[0]

energy_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')

def update(frame):
    for j in range(n):
        circle[j].center = par[j].sol_pos.x[frame], par[j].sol_pos.y[frame]
        line[j].set_xdata(par[j].sol_pos.x[:frame + 1])
        line[j].set_ydata(par[j].sol_pos.y[:frame + 1])
    energy_text.set_text(f'Total Energy: {Energy[frame]:.5f}')
    return circle + line + [energy_text]

ani = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)

plt.show() 
