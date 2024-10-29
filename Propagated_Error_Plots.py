# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 15:50:32 2018
This script generates two plots:
    1-How relative error varies with number of iterations (i.e: how the error accumulates)
    2-How the accumulated error varies with the paramter h
@author: Girish
"""
import Functions as F
import matplotlib.pyplot as plt

plt.close()
F.propagated_error()  #90 seconds to run
F.propagated_error_h()  #90 seconds to run
