import numpy as np
import math

AIR_DENSITY = 1.225 #air density in kg/m^3
g = 9.81    #gravity vector in m/s²
### A320 values ###
Cd0_land = 0.120  #zero-lift drag coefficient
k = 0.0334    #lift-induced drag coefficient factor
S = 122.6   #wing surface m²
weight = 60000  #mass in kg
Cst = 0.85  #friction coefficient
lift_coefficient = 0.1  #lift coefficient
angle_runaway = 0   #angle runway in °

def lift(v,S):  #compute the lift at instant t
    return lift_coefficient*S*(AIR_DENSITY*v**2/2)

def drag(k,Cd0,v,S): #compute the drag at instant t
    return drag_coefficient(k,Cd0)*S*(AIR_DENSITY*v**2/2)

def rolling_resistance_coefficient(v,Cst):  #compute the rolling resistance coefficient
    return (0.0041+0.000041*v)*Cst

def rolling_resistance(weight,v,Cst):   #compute the rolling_resistance force
    return (weight* rolling_resistance_coefficient(v,Cst))

def drag_coefficient(k,Cd0): #compute the drag coefficient
    return Cd0 + k*(lift_coefficient**2)

def slope_runaway(angle, weight):   #calculate the gravity force
    return weight*g*math.sin(angle)

def autobrakes(mode):   #applied the autobrake mode
    if mode == "MIN":
        return feet_in_meters(3)
    elif mode == "MED":
        return feet_in_meters(5)
    elif mode == "HIGH":
        return feet_in_meters(7)
    else:
        return feet_in_meters(11)

def feet_in_meters(feet):   #convert feet in meter
    return feet/3.2808

def calc_acc(v):    #compute acceleration
    acc = -drag(k,Cd0_land,v,S) - rolling_resistance(weight-lift(v,S),v,Cst) - slope_runaway(angle_runaway,weight-lift(v,S))
    acc = acc/weight - autobrakes("HIGH")*Cst
    return acc