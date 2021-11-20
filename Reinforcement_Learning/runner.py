import numpy as np
import time
import os
from collections import namedtuple
from math import e
from tensorforce import environments
from tensorforce.environments import environment
from LandingEnv import LandingEnvironment   #load the environnment
from LandingModel import LandingModel   #load the landing simulator

from tensorforce import Agent
from plot import plot_multiple

LandingModel = LandingModel()

environment = LandingEnvironment()

#creation of the agent initialize with the hyperparameters
agent = Agent.create(
        agent='ppo', environment=environment,
        # Automatically configured inputs and outputs of the network
        network=dict(
            type="auto",
            size=32,
            depth=4
        ),
        # Optimization
        batch_size=100, update_frequency=2, learning_rate=1e-3, subsampling_fraction=0.2,
        optimization_steps=5,
        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=1.0, estimate_terminal=False,
        #memory of the agnt
        memory=500000,
        # Critic
        critic_network='auto',
        critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
        # Preprocessing
        preprocessing=None,
        # Exploration
        exploration=0.01, variable_noise=0.0,
        # Regularization
        l2_regularization=0.1, entropy_regularization=0.1,
        # TensorFlow etc
        name='agent', device=None, parallel_interactions=1, seed=32, execution=None, saver=None,
        summarizer=None, recorder=None
    )

def run(environment, agent, n_episodes,nb_batch,test=False):    #run 1 batch of episode 

    Score = namedtuple("Score", ["reward","reward_mean","distance"])
    score = Score([],[],[])
    Episode = namedtuple("Episode",["rewards","decceleration"])

    # Loop over episodes
    for i in range(n_episodes):
        # Initialize episode
        episode = Episode([],[])
        episode_length = 0
        states = environment.reset()
        terminal = False
        reward = 0
        while not terminal:
            episode_length += 1
            if test:
                actions = agent.act(states=states,evaluation = True)
                
            # Run episode
            else:
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions,reward=reward)
                agent.observe(terminal=terminal, reward=reward)
            episode.rewards.append(reward)
            episode.decceleration.append(actions)

        rew = open("./data/"+str(nb_batch)+"/reward_"+str(i+1)+".txt",'w')
        rew.write(str(episode.rewards))
        rew.close()
        act = open("./data/"+str(nb_batch)+"/policy_"+str(i+1)+".txt",'w')
        act.write(str(episode.decceleration))
        act.close()
        print("Reward of the episode: {}".format(reward))
        score.reward.append(reward)
        score.reward_mean.append(np.mean(score.reward))
        score.distance.append(environment.LandingModel.Pos_vec[-1])
    
    return environment.LandingModel.Pos_vec[-1]

def create_folder(batchnumber):
    try:
        os.mkdir("./data/"+str(batchnumber))
    except:
        pass

def batch_information(i,num_bacth,temp_time,result_vec):    #print the batches information
    print("Batch {}/{}, Best result: {}, Time per batch {}s, Total ETA: {}mn{}s".format(
        i,num_bacth,int(result_vec[-1]),round(temp_time/i,1),
        round((temp_time*num_bacth/i)//60),
        round((temp_time * num_bacth / i) % 60)))

def trainer(environment, agent, n_episodes,n_episode_test=1):   #start the whole training of the model
    result_vec = [0]
    number_batches = round(n_episodes/100) + 1
    start_time = time.time()
    for i in range(1,number_batches+1):
        create_folder(i)
        temp_time = time.time() - start_time
        batch_information(i,number_batches-1,temp_time,result_vec)
        run(environment,agent,100,i)
        #result_vec.append(run(environment,agent,n_episode_test,test=True))
    agent.close()
    environment.close() 

#trainer(environment,agent,10000)
