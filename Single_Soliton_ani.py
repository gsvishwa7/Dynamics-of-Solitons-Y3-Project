# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 21:46:16 2018
This script generates the animation of a single soltion. 
@author: Girish
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

h = 0.05
dt = 0.0001
x = np.arange(-5, 15, h)
N = len(x)


def soliton_given(x, t=0, alpha=6):
    #intiial conditions
    return (12)*alpha**2*(np.cosh(alpha*(x-4*(alpha**2)*t)))**(-2)


u_initial=soliton_given(x, alpha=2)

# Preparing the plots
fig, ax = plt.subplots()
line, = ax.plot(x, u_initial,'b', label='Single Soliton')
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylim(bottom=-5)
plt.ylabel('u')


def init():
    line.set_ydata([np.nan]*N)
    return line,


def f(u):
    output = np.array([(-1/(4*h))*(u[(i+1)%N]**2 - u[(i-1)%N]**2) - (1/(2*h**3))*(u[(i+2)%N] - 2*u[(i+1)%N] + 2*u[(i-1)%N] - u[(i-2)%N]) for i in range(N)])
    return output   


def update(i):
    global u_final, u_initial
    #Implement RK4
    #Animation updates after every RK4 iteration by generating a new frame
    k1 = dt*f(u_initial)
    k2 = dt*(f(u_initial+k1/2))
    k3 = dt*(f(u_initial+k2/2))
    k4 = dt*(f(u_initial+k3))
    u_final = u_initial + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    line.set_ydata(u_final) 

    u_initial = u_final
    
    return line,

ani = animation.FuncAnimation(fig, update, init_func=init, frames=1500, repeat = 1, blit=True, interval=0)
plt.show()