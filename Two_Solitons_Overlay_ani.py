# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 22:58:16 2018
This script generated the animation for two interacting solitons, with an overlay 
of the same two solitons, but not interacting.
@author: Girish
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


h = 0.05
dt = 0.0001
x = np.arange(-5, 14, h)
N = len(x)


def soliton_given(x, t=0, alpha=3):
    #intiial conditions
    return (12)*alpha**2*(np.cosh(alpha*(x-4*(alpha**2)*t)))**(-2)


#Three initial functions: One for interacting two-soliton, the other two for the same solitons not interating
u_initial=soliton_given(x, alpha=3)+soliton_given(x,t=0.75,alpha=1.2)
u_initial2=soliton_given(x,alpha=3)
u_initial3=soliton_given(x,t=0.75,alpha=1.2)


# Preparing the plots
fig, ax = plt.subplots()
line, = ax.plot(x, u_initial,'b', label='2 Soliton Solution')
line2, = ax.plot(x, u_initial2,'g',label='Singe Soliton Solutions')
line3, = ax.plot(x, u_initial3,'g')
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel('u')



#Three animations, so three initialisation functions
def init():
    line.set_ydata([np.nan]*N)
    return line,

def init2():
    line2.set_ydata([np.nan]*N)
    return line2,

def init3():
    line3.set_ydata([np.nan]*N)
    return line3,
    

def f(u):
    #Define vector function f
    output = np.array([(-1/(4*h))*(u[(i+1)%N]**2 - u[(i-1)%N]**2) - (1/(2*h**3))*(u[(i+2)%N] - 2*u[(i+1)%N] + 2*u[(i-1)%N] - u[(i-2)%N]) for i in range(N)])
    return output   
   
    
#Three update functions for the three animations
#All three animations will generate new frames and plot them on the same figure

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
#    print(f'i is: \n {i}')
    return line,

def update2(j):
    
    global u_initial2, u_final2
    #Implement RK4
    k12 = dt*f(u_initial2)
    k22 = dt*(f(u_initial2+k12/2))
    k32 = dt*(f(u_initial2+k22/2))
    k42 = dt*(f(u_initial2+k32))
    u_final2 = u_initial2 + (1/6)*(k12 + 2*k22 + 2*k32 + k42)
    
    line2.set_ydata(u_final2)
    u_initial2 = u_final2
#    print(f'j is: \n {j}')
    return line2,

def update3(p):

    global u_initial3, u_final3
    #Implement RK4
    k13 = dt*f(u_initial3)
    k23 = dt*(f(u_initial3+k13/2))
    k33 = dt*(f(u_initial3+k23/2))
    k43 = dt*(f(u_initial3+k33))
    u_final3 = u_initial3 + (1/6)*(k13 + 2*k23 + 2*k33 + k43)
    
    line3.set_ydata(u_final3)
    u_initial3 = u_final3
    
#    print(f'p is: \n {p}')
    
    return line3,


ani = animation.FuncAnimation(fig, update, init_func=init, frames = 1, repeat = 1, blit=False, interval=0)
ani2 = animation.FuncAnimation(fig, update2, init_func=init2, frames = 1, repeat = 1, blit=False, interval=0)
ani3 = animation.FuncAnimation(fig, update3, init_func=init3, frames = 1, repeat = 1, blit=False, interval=0)

plt.show()
