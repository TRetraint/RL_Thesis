import numpy as np
from tensorforce.environments import Environment
from LandingModel import LandingModel #import the Landing Simulator

#class representing the environnement
class LandingEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.LandingModel = LandingModel()  #create an instance of the Landing Simulator
        self.NUM_ACTIONS = len(self.LandingModel.action_vec)    #size of the vector of actions
        self.STATES_SIZE = len(self.LandingModel.states_vec)    #seize of the vector of states
        self.max_step_per_episode = 1000    #represent the number of maximum time step of an episode
        self.finished = False   #variable saying if the plane manage to successfully land
        self.episode_end = False    #variable saying if the plane went above the runway

    #States function
    def states(self):
        return dict(type="float", shape=(self.STATES_SIZE,))
    
    #Actions function
    def actions(self):
        return dict(type="int",num_values=self.NUM_ACTIONS)
    
    #return the max timesteps of an episode
    def max_episode_timesteps(self):
        return self.max_step_per_episode
    
    #close the environnement at then end of the learning process
    def close(self):
        return super().close()
    
    #reset function, gives the original state of an episode
    def reset(self):
        state = np.array([-0.73263,np.random.normal(69.37,4.32),267.2,self.LandingModel.safe_breaking_dist])    #original state
        self.LandingModel = LandingModel()  #new instance of the landing model
        return state    #return the original state
    
    #Compute at each timestep the actions of the model, update the state, and calculate the reward
    def execute(self, actions,reward):
        next_state = self.LandingModel.compute_timestep(action=actions) #compute the new states based on the action taken
        terminal = self.terminal()  #check if an episode is done
        reward = reward + self.reward(actions)  #compute the reward
        return next_state, terminal, reward

    #Terminal function, test if an episode is done or not
    def terminal(self):
        self.finished = self.LandingModel.V_vec[-1] <= 0    #test if the plane manage to successfully land
        self.episode_end = self.LandingModel.Pos_vec[-1] >= self.LandingModel.max_braking_distance  #test if the plane ran off the runway
        return self.finished or self.episode_end    #return the result

    def reward(self,actions):
        const_rew = 500     #constant reward if the plane manage to land (gamma coefficient)
        coef_rew_corr = 0.01    #Beta coefficient

        const_pun = -1000   #constant punishement (theta coefficient)
        coef_pun_corr = 1.0 #delta coefficient

        #determination of the alpha coefficient
        if actions == 0:
            coef = 10
        elif actions == 1:
            coef = 8
        elif actions == 2:
            coef = 5
        elif actions == 3:
            coef = 2
        elif actions == 4:
            coef = 1

        #compute the reward
        reward = -(self.LandingModel.V_vec[-1] - self.LandingModel.V_vec[-2])*coef  #first line

        if self.finished:
            reward += const_rew - ((self.LandingModel.safe_breaking_dist - self.LandingModel.Pos_vec[-1])**2)*coef_rew_corr #2nd line
        
        if self.episode_end:
            reward += const_pun - self.LandingModel.V_vec[-1]*coef_pun_corr #3rd line
        
        return reward




        