# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 15:43:32 2018
This script generates a heat map depicting accuracy and stability of the
numerical single soliton solution with alpha=1.5, after 150 RK4 iterations.
The heat maps in the repot use alpha=1.5 and alpha=3, after 500 or 1000 iterations.
@author: Girish
"""
import Functions as F
import matplotlib.pyplot as plt

plt.close()

F.error_finder(alpha_val=1.5,iterations=150) #75-90 seconds to run

#If the user does not wish to have this plot generated, the function can be commented out.
F.extract_stability_boundary()  #fast 
