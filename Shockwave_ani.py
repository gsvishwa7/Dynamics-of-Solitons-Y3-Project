# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:29:50 2018
This script generates an undamped shockwave starting with a soliton solution 
as the initial condition
@author: Girish
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


h = 0.05
dt = 0.0001
x = np.arange(-5, 15, h)
N = len(x)


def soliton_given(x, t=0, alpha=3):
    #intiial conditions
    return (12)*alpha**2*(np.cosh(alpha*(x-4*(alpha**2)*t)))**(-2)


u_initial=soliton_given(x, alpha=3)

# Preparing the plots
fig, ax = plt.subplots()
line, = ax.plot(x, u_initial,'b', label='Shock Wave')
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylim(bottom=-10)
plt.ylabel('u')


def init():
    line.set_ydata([np.nan]*N)
    return line,


def f(u):
    output = np.zeros([N])
    for i in range(N):
        output[i] = (-1/(4*h))*(u[(i+1)%N]**2 - u[(i-1)%N]**2)
    return output   


def update(i):
    global u_final, u_initial

    #Implement RK4
    k1 = dt*f(u_initial)
    k2 = dt*(f(u_initial+k1/2))
    k3 = dt*(f(u_initial+k2/2))
    k4 = dt*(f(u_initial+k3))
    
    u_final = u_initial + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    line.set_ydata(u_final) 

    u_initial = u_final

    return line,

ani = animation.FuncAnimation(fig, update, init_func=init, frames=9500, repeat = True, blit=True, interval=0)
plt.show()
