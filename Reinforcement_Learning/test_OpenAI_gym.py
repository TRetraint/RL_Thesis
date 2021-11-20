from tensorforce import Runner
from tensorforce.execution import runner
from tensorforce import Agent

def main():
    # OpenAI-Gym environment specification
    environment = dict(environment='gym', level='CartPole-v1',visualize=True)
    # or: environment = Environment.create(
    #         environment='gym', level='CartPole-v1', max_episode_timesteps=500)

    # PPO agent specification
    agent = dict(
        agent='ppo',
        # Automatically configured network
        network='auto',
        # PPO optimization parameters
        batch_size=10, update_frequency=2, learning_rate=3e-4,
        subsampling_fraction=0.33,
        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.99,
        # Baseline network and optimizer
        critic_network=dict(type='auto', size=32, depth=1),
        critic_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Default additional config values
        config=None,
    )
    # or: Agent.create(agent='ppo', environment=environment, ...)
    # with additional argument "environment" and, if applicable, "parallel_interactions"

    # Initialize the runner
    runner = Runner(agent=agent, environment=environment, max_episode_timesteps=500)

    # Train for 200 episodes
    runner.run(num_episodes=200)
    runner.close()

    # plus agent.close() and environment.close() if created separately


def test():
    environment = dict(environment='gym', level='CartPole-v1')
    agent = Agent.create(agent='agent.json',environment=environment)
    

if __name__ == '__main__':
    main()