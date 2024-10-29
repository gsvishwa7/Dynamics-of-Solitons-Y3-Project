# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 16:11:51 2018
This script generates plots for the momentum and energy of:
    1. Two colliding/interacting solitons
    2. A single soliton
Alpha values can be changed if desired. 
For the 2-soliton case, the user is advised to choose parameters such that the
two solitons actually do interact within the number of iterations performed.
@author: Girish
"""
import Functions as F
import matplotlib.pyplot as plt

plt.close()

F.momentum_and_energy(alpha_1=3, alpha_2=1.3, t=0.6) #10-20 seconds to run
F.check_single_soliton(alpha_value=3) #<10 seconds to run
