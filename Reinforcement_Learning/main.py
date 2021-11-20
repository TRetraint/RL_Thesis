#import libraries
from tensorforce import environments
from tensorforce.agents import agent
from tensorforce.environments import environment
from LandingEnv import LandingEnvironment
from LandingModel import LandingModel
from tensorforce import Agent
from runner import trainer

environment = LandingEnvironment() #import the environement

def main():
    #creation of the optimized agent
    agent = Agent.create(
        agent='ppo', environment=environment,
        # Automatically configured network
        network=dict(
            type="auto",
            size=64,
            depth=4
        ),
        # Optimization
        batch_size=100, update_frequency=2, learning_rate=1e-3, subsampling_fraction=0.3,
        optimization_steps=100,
        # Reward estimation
        likelihood_ratio_clipping=0.1, discount=1.0, estimate_terminal=False,
        # Critic
        memory=500000,
        critic_network='auto',
        critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
        # Preprocessing
        preprocessing=None,
        # Exploration
        exploration=0.001, variable_noise=0.0,
        # Regularization
        l2_regularization=0.001, entropy_regularization=0.01,
        # TensorFlow etc
        name='best_agent', device=None, parallel_interactions=3, seed=40, execution=None,
        saver=dict(directory="./best_model/", filename="best_agent"),
        recorder=dict(directory="./best_model/", frequency=1000)
    )
    trainer(environment,agent,10000)    #start the training over 10000 landings
main()