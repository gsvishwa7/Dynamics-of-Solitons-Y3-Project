# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 22:00:14 2018
This script generates the animation of a single soliton, with an overlay of the analytic solution.
@author: f18ho
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


h = 0.05
dt = 0.0001
x = np.arange(-5, 15, h)
N = len(x)
t_i=0

def soliton_given(x, t=0, alpha=3):
    #intiial conditions
    return (12)*alpha**2*(np.cosh(alpha*(x-4*(alpha**2)*t)))**(-2)
    
    

u_initial=soliton_given(x, alpha=3)
u_initial_2=soliton_given(x, alpha=3)
# Preparing the plots
fig, ax = plt.subplots()
line, = ax.plot(x, u_initial,'b', label='Single Soliton')
line_2, = ax.plot(x, u_initial_2, 'g',label='Analytic Soliton')
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylim(bottom=-10)
plt.ylabel('u')



def init():
    line.set_ydata([np.nan]*N)
    return line,


def init_2():
    line_2.set_ydata([np.nan]*N)
    return line_2,



def f(u):
    #Define vector function f
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



def update_2(i):
    global u_final_2, u_initial_2, t_i
    #Propagating the analytic solution by changing t
    u_final_2=soliton_given(x,t_i+dt,alpha=3)
    t_i+=dt

    line_2.set_ydata(u_final_2) 
    return line_2,



ani = animation.FuncAnimation(fig, update, init_func=init, frames=3000, repeat = 0, blit=False, interval=0)
ani_2 = animation.FuncAnimation(fig, update_2, init_func=init_2, frames=3000, repeat = 0, blit=False, interval=0)

plt.show()
