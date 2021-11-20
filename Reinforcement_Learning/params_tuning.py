#import libraries
import os
import itertools
import shutil
import numpy as np
from collections import namedtuple
from runner import trainer
from tensorforce import Agent
from LandingEnv import LandingEnvironment
from LandingModel import LandingModel

LandingModel = LandingModel()   #import Landing Simulator

environment = LandingEnvironment()  #import the environement

def create_agent(param_grid, i, directory, environment):    #create the agent for the According to the Grid Search parameters
    return Agent.create(
        agent="ppo",
        environment=environment,
        # Automatically configured network
        network=dict(
            type=param_grid["network"],
            size=param_grid["size"],
            depth=param_grid["depth"],
        ),
        # Optimization
        batch_size=param_grid["batch_size"],
        update_frequency=param_grid["update_frequency"],
        learning_rate=param_grid["learning_rate"],
        subsampling_fraction=param_grid["subsampling_fraction"],
        optimization_steps=param_grid["optimization_steps"],
        # Reward estimation
        likelihood_ratio_clipping=param_grid["likelihood_ratio_clipping"],
        discount=param_grid["discount"],
        estimate_terminal=param_grid["estimate_terminal"],
        # Critic
        critic_network="auto",
        critic_optimizer=dict(
            optimizer="adam",
            multi_step=param_grid["multi_step"],
            learning_rate=param_grid["learning_rate_critic"],
        ),
        # Preprocessing
        preprocessing=None,
        # Exploration
        exploration=param_grid["exploration"],
        variable_noise=param_grid["variable_noise"],
        # Regularization
        l2_regularization=param_grid["l2_regularization"],
        entropy_regularization=param_grid["entropy_regularization"],
        # TensorFlow etc
        name="agent_" + str(i),
        device=None,
        parallel_interactions=5,
        seed=40,
        execution=None,
        recorder=dict(directory=directory, frequency=1000),
        summarizer=None,
        saver=dict(directory=directory, filename="agent_" + str(i)),
    )


def gridsearch_tensorforce(environment, param_grid_list, n_episodes):   #Start the Grid Search to tune the hyperparameters
    GridSearch = namedtuple("GridSearch", ["scores", "names"])
    gridsearch = GridSearch([], [])

    # Compute the different parameters combinations
    param_combinations = list(itertools.product(*param_grid_list.values()))
    for i, params in enumerate(param_combinations, 1):
        if not os.path.exists(os.path.join("Grid_Search", "Graphs", str(i))):
            os.mkdir(os.path.join("Grid_Search", "Graphs", str(i)))
        # fill param dict with params
        param_grid = {
            param_name: params[param_index]
            for param_index, param_name in enumerate(param_grid_list)
        }
        directory = os.path.join(os.getcwd(), "Grid_Search", "Models", str(i))
        if os.path.exists(directory):
            shutil.rmtree(directory, ignore_errors=True)

        agent = create_agent(param_grid, i, directory, environment)
        # agent = Agent.load(directory="data/checkpoints")
        gridsearch.scores.append(
            trainer(
                environment,
                agent,
                n_episodes,
            )
        )
        #store_results_and_graphs(i, environment, param_grid)
        gridsearch.names.append(str(param_grid))
    dict_scores = dict(zip(gridsearch.names, gridsearch.scores))
    best_model = min(dict_scores, key=dict_scores.get)
    print(
        "best model",
        best_model,
        "number",
        np.argmin(gridsearch.scores),
        "score",
        dict_scores[best_model],
    )

#Parameters of the Grid Search 
param_grid_list = {}
param_grid_list["PPO"]={
        "batch_size": [100],
        "update_frequency": [20],
        "learning_rate": [1e-3],
        "subsampling_fraction": [0.1, 0.2, 0.3],
        "optimization_steps": [100],
        "likelihood_ratio_clipping": [0.1, 0.2, 0.3],
        "discount": [0.5, 0.99, 1.0],
        "estimate_terminal": [False],
        "multi_step": [30],
        "learning_rate_critic": [1e-3],
        "exploration": [0.01],
        "variable_noise": [0.0],
        "l2_regularization": [0.1, 0.01, 0.001],
        "entropy_regularization": [0.1, 0.01, 0.001],
        "network": ["auto"],
        "size": [32, 64, 128],
        "depth": [2, 4, 6 , 8],
}

gridsearch_tensorforce(environment,param_grid_list["PPO"],10000)