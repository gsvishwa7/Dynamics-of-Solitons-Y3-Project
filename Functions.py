# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:42:15 2018
This script contains all functions that do not render animations.
The functions are used primarily for numerical analysis of the RK4 method.
@author: Girish
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def soliton_given(x, t=0, alpha=3):
    #intiial conditions
    return (12)*alpha**2*(np.cosh(alpha*(x-4*(alpha**2)*t)))**(-2)



def f(u,h,N):
    #Define vector function f
    output = np.array([(-1/(4*h))*(u[(i+1)%N]**2 - u[(i-1)%N]**2) - (1/(2*h**3))*(u[(i+2)%N] - 2*u[(i+1)%N] + 2*u[(i-1)%N] - u[(i-2)%N]) for i in range(N)])
    return output 



def momentum_and_energy(alpha_1=3, alpha_2=1.2, t=0.6):
    #Measures momentum and energy of two solitons over 2500 iterations
    #Alpha values of the two solitons can be given as parameters, along with initial separation of the solitons
    #The user may need to change the number of iterations if the given parameters are changed
    #This is to ensure that the two solitons actually interact!
    
    h, dt = 0.05, 0.0001
    x = np.arange(-5, 20, h)
    N = len(x)
    iterations=2500
    
    #Two solitons well separated as initial condition
    u_initial=soliton_given(x, alpha=alpha_1)+soliton_given(x,t=t,alpha=alpha_2)
    
    #Initial momentum and energy of two-soliton
    momentum=np.array([np.sum(u_initial)])
    energy=np.array([0.5*np.sum((u_initial)**2)])
    
    for j in range(iterations):
        #RK4
        k11 = dt*f(u_initial,h,N)
        k21 = dt*(f(u_initial+k11/2,h,N))
        k31 = dt*(f(u_initial+k21/2,h,N))
        k41 = dt*(f(u_initial+k31,h,N))
        u_final = u_initial + (1/6)*(k11 + 2*k21 + 2*k31 + k41)
        momentum=np.append(momentum, np.sum(u_final))
        energy=np.append(energy, (1/2)*np.sum(u_final**2))
        u_initial = u_final
        
    plt.figure(1)
    plt.plot(np.linspace(0,iterations,iterations+1), momentum, 'r', label='Momentum of Interacting 2-Soliton')
    plt.xlabel('Time')
    plt.ylabel('Momentum')
    plt.legend(loc='upper left')
    plt.show
    
    plt.figure(2)
    plt.plot(np.linspace(0,iterations,iterations+1), energy, 'b', label='Energy of Interacting Solitons')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.legend(loc='upper left')
    plt.show
    


def error_finder(alpha_val=3.0,iterations=500):
    #This function generates a heat map of accuracy and stability 
    #It catches overflows and interprets it as an unstable solution
    #If no overflow, then the function measures the relative error of the solution
    
    np.seterr(all="raise")
    
    i=0
    global N_h, N_dt    #In the report, N_h=20 and N_dt=30 and iterations = 500 or 1000
    N_h=8
    N_dt=10
    global stability_val
    stability_val=np.zeros([N_h*N_dt])
    global h_range, dt_range
    h_range=np.linspace(0.01,0.2,N_h)[::-1]
    dt_range=np.linspace(0.0000001,0.001,N_dt)

    for h in h_range:
        x = np.arange(-5, 60, h)
        N=len(x)
        for dt in dt_range:
            #stability=True
            u_initial=soliton_given(x, alpha=alpha_val)
            for j in range(iterations):
                try:
                    #print(dt, h)
                    #global u_final, u_initial
                    stability=True
                    #Implement RK4
                    k1 = dt*f(u_initial,h,N)
                    k2 = dt*(f(u_initial+k1/2,h,N))
                    k3 = dt*(f(u_initial+k2/2,h,N))
                    k4 = dt*(f(u_initial+k3,h,N))
                    u_final = u_initial + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
                    u_initial=u_final
                except FloatingPointError:
 
                    stability=False
                    break
           
           #Check accuracy if solution is stable
           #Else, assign its accuracy value as np.NaN
            if stability==True:
                analytic = soliton_given(x, t=iterations*dt, alpha=alpha_val)
                dev_percent=np.sqrt(np.sum((u_final - analytic)**2)/N)/np.amax(analytic)
                stability_val[i]=dev_percent
            else:
                stability_val[i]=np.NaN
            
            i+=1
    stability_val_r=np.reshape(stability_val, (N_h,N_dt))        
    
    fig, ax = plt.subplots()
    im = ax.imshow(stability_val_r)
    
        # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    #Show and label all ticks
    ax.set_xticks(np.arange(N_dt))
    ax.set_yticks(np.arange(N_h))
    
    xlabels = [f"{dt:.3E}" for dt in dt_range]
    ylabels = [f"{h:.3E}" for h in h_range]
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    ax.set_xlabel(r'$dt$')
    ax.set_ylabel(r'$h$')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.title(r'$h$ and $dt$ Dependence of Stability and Accuracy')
    plt.show()
    

#Fitting functions used for plots
def fit_func(x,a,b):
    return a*x**(1/3)+b*x

def fit_func2(x,a):
    return a*(x**2)


    
def extract_stability_boundary():
    #This function extracts the stability boundary values from the error_finder function
    #The error_finder function needs to be run first
    #So that there will be h and dt values to extract a boundary from in the first place!
    
    boundary_val_h, boundary_val_dt = [], []
    
    for i in range(0,N_h*N_dt,N_dt):
        h=h_range[i//N_dt]
        for j in range(N_dt):
            if np.isnan(stability_val[i+j])==True:
                boundary_val_h.append(h)
                boundary_val_dt.append(dt_range[j-1])
                break
            
    popt, pcov = curve_fit(fit_func, np.array(boundary_val_dt), np.array(boundary_val_h))
    print(f'a = {popt[0]}, b={popt[1]}')
    
    plt.figure(2)
    plt.plot(np.linspace(0,0.001,101), fit_func(np.linspace(0,0.001,101),popt[0],popt[1]), 'r', label=r'$h=a(dt)^{1/3}+b(dt)$')
    plt.scatter(boundary_val_dt, boundary_val_h,label='Stability Boundary Points')
    
    plt.xlim(-0.0001,0.001)
    plt.title('Stability boundary values')
    plt.xlabel(r'$dt$')
    plt.ylabel(r'$h$')
    plt.legend(loc='upper left')
    
    plt.show()
    


def initial_cosine(x,N):
    #initial condition for wave breaking function
    #The width of the cosine can be changed if needed
    output = np.zeros([N])
    for i in range(N):
        if -30 <= x[i] <= 30:
            output[i] =  10*np.cos(x[i]*np.pi/60)            
    return output



def wavebreak():
    #Generates 6 subplots which show the progression of wavebreaking over time
    h = 0.1
    dt = 0.001
    x = np.arange(-30, 150, h)
    N = len(x)
    
    #Creat subplots
    fig, axs = plt.subplots(6,1,sharex=False,sharey=False)
    u_initial=initial_cosine(x,N)
    
    #Create first subplot showing the initial waveform
    axs[0].plot(x,u_initial,color='#8733FF')
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["left"].set_visible(False)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_ylim(bottom=0)
    axs[0].set_ylabel('0',rotation=30, rotation_mode='anchor')
    fig.text(0.5, 0.02, r'$x$', ha='center')
    fig.text(0.02, 0.5, 'No. of Iterations', va='center', rotation='vertical')
    fig.suptitle('Wave Breaking of Initial Cosine Waveform')

    for i in range(5):
        #Each of new subplot shows what has happened after 2500 iterations
        for j in range(2500*i,2500*(i+1)):
            k1 = dt*f(u_initial,h,N)
            k2 = dt*(f(u_initial+k1/2,h,N))
            k3 = dt*(f(u_initial+k2/2,h,N))
            k4 = dt*(f(u_initial+k3,h,N))
            u_final = u_initial + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
            u_initial = u_final
        axs[i+1].plot(x,u_final,color='#8733FF')
        axs[i+1].set_ylabel(f'{(i+1)*2500}',rotation=30, rotation_mode='anchor')
        axs[i+1].set_ylim(bottom=0)
        axs[i+1].spines["top"].set_visible(False)
        axs[i+1].spines["right"].set_visible(False)
        axs[i+1].spines["left"].set_visible(False)

        
        if i != 4:
            axs[i+1].set_xticks([])
            axs[i+1].set_yticks([])
        else:
            axs[i+1].set_yticks([])
            axs[i+1].set_xticks(np.linspace(-30,150,10))
            axs[i+1].set_xticklabels(np.linspace(-30,150,10))
    plt.show()



def propagated_error():
    #This function measures the error accumulated after each iteration, for 1500 iterations
    #In the report, iterations = 5000 
    h, dt = 0.05, 0.0001
    x = np.arange(-7, 50, h)
    N=len(x)
    alpha_range=[1,1.5,2,3]
    for alpha_val in alpha_range:
        u_initial=soliton_given(x, alpha=alpha_val)
        dev_percent=0
        error_after_sometime=[0]
        iterations=1500
        for j in range(iterations):
            #RK4
            k1 = dt*f(u_initial,h,N)
            k2 = dt*(f(u_initial+k1/2,h,N))
            k3 = dt*(f(u_initial+k2/2,h,N))
            k4 = dt*(f(u_initial+k3,h,N))
            u_final = u_initial + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
            u_initial=u_final
            analytic = soliton_given(x, t=(j+1)*dt, alpha=alpha_val)
            dev_percent=np.sqrt(np.sum((u_final - analytic)**2)/N)/np.amax(analytic)
            error_after_sometime.append(dev_percent)

            
        plt.plot(np.linspace(0,iterations*dt,iterations+1),error_after_sometime,label=r'$\alpha =$' +f'{alpha_val}')
        plt.xlabel('t')
        plt.ylabel('Relative Error')
        plt.title('Error Accumulated Over Time')
        plt.legend(loc='upper left')
    plt.show()
    
    
    
def propagated_error_h(alpha_val=2.0):
    #This function measures the error after 1200 iterations, for different h values
    #In the report, iterations = 5000 and 16 h values were used instead of 10
    dt = 0.00001
    h_vals = np.linspace(0.025,0.1,10)
    error=[]
    for h in h_vals:
        x = np.arange(-7, 20, h)
        N=len(x)
        dev_percent=0
        u_initial=soliton_given(x, alpha=alpha_val)
        iterations=1200
        for j in range(iterations):
            #RK4
            k1 = dt*f(u_initial,h,N)
            k2 = dt*(f(u_initial+k1/2,h,N))
            k3 = dt*(f(u_initial+k2/2,h,N))
            k4 = dt*(f(u_initial+k3,h,N))
            u_final = u_initial + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
            u_initial=u_final
            analytic = soliton_given(x, t=(j+1)*dt, alpha=alpha_val)
            
        dev_percent=np.sqrt(np.sum((u_final - analytic)**2)/N)/np.amax(analytic)
        error.append(dev_percent) 
            
    popt, pcov = curve_fit(fit_func2, h_vals, np.array(error))
    print(popt)
    plt.figure(2)
    plt.plot(h_vals,fit_func2(h_vals,popt),'r',label=r'$h^{2}$ fit')
    plt.plot(h_vals,error,label='Measured Error')
    plt.xlabel(r'$h$')
    plt.ylabel('Relative Error')
    plt.title(r'Error Accumulated After 5000 Iterations vs $h$ value')
    plt.legend(loc='upper left')
    plt.show()



def speed(number=10):
    #This function measures soliton speed and generates a plot showing... 
    #...the variation of soliton speed with alpha
    #In the report, number = 100
    
    h, dt = 0.04, 0.00005
    x = np.arange(-5, 15, h)
    N = len(x)
    speed=np.zeros([number])
    a=0
    for alpha_val in np.linspace(1,10,number):
        u_i = soliton_given(x, t=0, alpha=alpha_val)
        old_pos=x[np.argmax(u_i)]
        d=0
        for j in range(500):
            #RK4
            k1 = dt*f(u_i,h,N)
            k2 = dt*(f(u_i+k1/2,h,N))
            k3 = dt*(f(u_i+k2/2,h,N))
            k4 = dt*(f(u_i+k3,h,N))
            u_f = u_i + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
            u_i = u_f
            
        #Checking how far the peak of the soliton has travelled     
        b=np.argmax(u_f)
        d=x[b]-old_pos
        speed[a]=d/(500*dt)
        a+=1
        
    print(f'The speed is: {speed}')
    plt.figure(2)
    plt.scatter((np.linspace(1,10,len(speed))),speed,c='g',marker="x",label='Calculated Values')
    plt.plot(np.linspace(1,10,len(speed)),np.array([4*c**2 for c in np.linspace(1,10,len(speed))]),'r',label='Theoretical Values')
    plt.title(r'Speed of Soliton vs $\alpha$ value')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Speed')
    plt.legend(loc='upper left')
    
def check_single_soliton(alpha_value=3):
    #Identitical to the momentum_and_energy function but for a single soliton
    #Not a very important function, but was added later in the project as an extra
    
    h, dt = 0.05, 0.0001
    x = np.arange(-5, 15, h)
    N = len(x)
    iterations=2000
    
    u_initial=soliton_given(x, alpha=alpha_value)
    momentum=np.array([np.sum(u_initial)])
    energy=np.array([0.5*np.sum((u_initial)**2)])
    
    for j in range(iterations):
        k11 = dt*f(u_initial,h,N)
        k21 = dt*(f(u_initial+k11/2,h,N))
        k31 = dt*(f(u_initial+k21/2,h,N))
        k41 = dt*(f(u_initial+k31,h,N))
        u_final = u_initial + (1/6)*(k11 + 2*k21 + 2*k31 + k41) 
        momentum=np.append(momentum, np.sum(u_final))
        energy=np.append(energy, (1/2)*np.sum(u_final**2))
        u_initial = u_final
        
    plt.figure(3)   
    plt.plot(np.linspace(0,iterations,iterations+1), momentum, 'r', label='Momentum of Single Soliton')
    plt.xlabel('Time')
    plt.ylabel('Momentum')
    #plt.ylim(0,4000)
    plt.legend(loc='upper left')
    
    plt.figure(4)
    plt.plot(np.linspace(0,iterations,iterations+1), energy, 'b', label='Energy of Single Soliton')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.legend(loc='upper left')
    plt.show
    
#error_finder(alpha_val=1.5,iterations=150) #75-90 seconds    
#extract_stability_boundary()  #fast  

#propagated_error()  #90 seconds
#propagated_error_h()  #90 seconds
    
#speed() #20-30 seconds
#momentum_and_energy(alpha_1=3, alpha_2=1.3, t=0.6) #10-20 seconds
#check_single_soliton() #<10 seconds
#wavebreak() #3-4 minutes