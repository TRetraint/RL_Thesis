import numpy as np
import math

class LandingModel:
    def __init__(self):
        ###  A320 Values ###
        self.g = 9.81 #gravity vector in m/s²
        self.m = 60000 #mass in kg
        self.S = 122.6 #wings surface in m²
        self.RHO = 1.225 #air density in kg/m^3
        self.CD0_LAND = 0.120 #zero-lift drag coefficient
        self.k = 0.0334 #lift-induced drag coefficient factor
        self.Cst = 1 #friction coefficient
        self.lift_coef = 0.1 #lift coefficient
        self.slope = 0 #runaway slope in °
        self.tyre_condtition = 1 #coefficient represneting tyres conditions
        self.safe_breaking_dist = 1500 #safe braking distance parameter
        self.security_coefficient = 1.15 #coefficient to determine the lenght of the maximal acceptable braking distance
        self.max_braking_distance = self.security_coefficient*self.safe_breaking_dist
        self.wind_speed = 0

        self.timestep = 0.1 #value of the time between each states (0.1s)

        self.action_vec = np.array([0,1,2,3,4]) #Action Ensemble that the agent can take

        self.Acc_vec = [-0.73263]   #initilize the value of the deceleration at t=0
        self.V_vec = [np.random.normal(69.37,4.32)]    #initilize the speed of the plane at t=0
        self.Pos_vec = [267.2]  #initilize the x position of the plane at t=0 (landing start when the plane reach 50ft altitude, the compute only concerne the ground phase)

        self.states_vec = np.array([self.Acc_vec[-1],self.V_vec[-1],self.Pos_vec[-1],self.safe_breaking_dist])  #State of the environnement
        
      
    def calc_lift(self,v): #Calculation of the lift of the plane
        return self.lift_coef*self.S*(self.RHO*(v+self.wind_speed)**2/2)

    def calc_drag(self,v):  #Calculation of the drage of the plane
        return self.calc_drag_coefficient()*self.S*(self.RHO*(v*self.wind_speed)**2/2)
    
    def calc_drag_coefficient(self):    #calculation of the drag coefficient
        return self.CD0_LAND + self.k*(self.lift_coef**2)
    
    def rolling_resistance_coeffcient(self,v):  #calculation of the rolling resistance coefficient
        return (0.0041+0.000041*v)*self.Cst
    
    def calc_rolling_resistance_force(self,v,m):    #Calculation of the rolling resistance force
        return m*self.rolling_resistance_coeffcient(v)
    
    def calc_slope_runaway_force(self,m):   #Calculation of the action of the runway on the plane
        return m*self.g*math.sin(self.slope)
    
    
    def calc_acc(self,v):   #Compute all the forces to calculate the acceleration of the plane
        acc = -self.calc_drag(v) - self.calc_rolling_resistance_force(v,self.m-self.calc_lift(v)) - self.calc_slope_runaway_force(self.m - self.calc_lift(v))
        acc = acc/self.m
        return acc

    def compute_timestep(self,action):  #Compute the action of the model and update the state of the plane for the next timestep
        acc_sum = 0
        #translate the action into the value of the deceleration
        if action == 0:
            acc_sum = 0
        elif action == 1:
            acc_sum = -0.914411
        elif action == 2:
            acc_sum = -1.524018
        elif action == 3:
            acc_sum = -2.1336259
        else:
            acc_sum = -3.3528407
        
        acc_sum = acc_sum*self.tyre_condtition*self.Cst #apply the coefficients of friction and tyre condition the the deceleration

        x = self.Pos_vec[-1] + self.V_vec[-1]*self.timestep #compute the new x position of the plane
        a = self.calc_acc(self.V_vec[-1]) + acc_sum #compute the new acceleration
        v = self.V_vec[-1] + a*self.timestep    #compute the new velocity of the plane

        #add the value to the respective lists
        self.Pos_vec.append(x)
        self.Acc_vec.append(a)
        self.V_vec.append(v)

        return np.array([self.Acc_vec[-1],self.V_vec[-1],self.Pos_vec[-1],self.safe_breaking_dist]) #return the new state


