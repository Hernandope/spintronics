import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits import mplot3d

def eq1(x, y): 
    return ((x - y)/2) 

# A sample differential equation "dy/dx = -y" 
def eq2(x, y): 
    return (-y) 

# differential equation for precision term of LLGS" 
def eq3(t, M):
    return -(gamma*mu_0)/(1+alpha**2)*np.cross(M,H)
    
# # Finds value of y for a given x using step size h 
# # and initial value y0 at x0. 
def rungeKuttaNew(diff_eq, x_n, y_n, x, h): 
    y = y_n 
    n = int((x - x_n)/h) 
    # Iterate for n number of iterations 
    for i in range(1, n + 1): 
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = h * diff_eq(x_n, y) 
        k2 = h * diff_eq(x_n + 0.5*h, y + 0.5*k1*h) 
        k3 = h * diff_eq(x_n + 0.5*h, y + 0.5*k2*h) 
        k4 = h * diff_eq(x_n + h, y + k3*h) 
        # Update next value of y 
        y = y + (1.0 / 6.0)*(k1 + 2*k2 + 2*k3 + k4) 
        # Update next value of x 
        x_n = x_n + h 
    return y 

def rungeKutta(diff_eq, x_n, y_n, x, h): 
    y = y_n 
    n = int((x - x_n)/h) 
    # Iterate for n number of iterations 
    for i in range(1, n + 1): 
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = h * diff_eq(x_n, y) 
        k2 = h * diff_eq(x_n + 0.5*h, y + 0.5*k1) 
        k3 = h * diff_eq(x_n + 0.5*h, y + 0.5*k2) 
        k4 = h * diff_eq(x_n + h, y + k3) 
        # Update next value of y 
        y = y + (1.0 / 6.0)*(k1 + 2*k2 + 2*k3 + k4) 
        # Update next value of x 
        x_n = x_n + h 
    return y 