import argparse
import ast
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tf_agents.policies import PolicySaver
from Util import execute_tf
from LinkGenerationEnv import LinkGenerationEnvV2_Continuous
import tensorflow as tf
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
import reverb

import gym

import warnings

warnings.filterwarnings("ignore")


def collect_episode(environment, policy, num_episodes, rb_observer):
    driver = py_driver.PyDriver(
        environment,
        py_tf_eager_policy.PyTFEagerPolicy(
            policy, use_tf_function=True),
        [rb_observer],
        max_episodes=num_episodes)
    initial_time_step = environment.reset()
    driver.run(initial_time_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_links", default=4, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--decay_rate", default=0.2, type=float)
    parser.add_argument("--tradeoff", default=0.7, type=float)
    parser.add_argument("--actions", default=[0.4, 0.7])
    parser.add_argument("--num_iterations", default=5000, type=int)
    parser.add_argument("--replay_buffer_capacity", default=10000, type=int)
    parser.add_argument("--fc_layer_params", default=(256, 256))
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--gamma", default=1, type=float)
    parser.add_argument("--eval_interval", default=50000, type=int)
    parser.add_argument("--num_eval_episodes", default=20, type=int)
    parser.add_argument("--collect_episodes_per_iteration", default=1, type=int)

    args = parser.parse_args()

    env_name = "LinkGenerationEnv"
    file_name = f"{args.n_links}links_continuous_actions"
    print("---------------------------------------")
    print(f"Links:{args.n_links}, Actions:{args.actions}")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    gym.envs.register(
        id='LinkGenerationEnv',
        entry_point='__main__:LinkGenerationEnvV2_Continuous',
        max_episode_steps=500,
        kwargs={'n_links': args.n_links,
                'threshold': args.threshold,
                'decay_rate': args.decay_rate,
                'actions': args.actions,
                'tradeoff': args.tradeoff,
                'remove_hopeless_links': False}
    )

    env_name = "LinkGenerationEnv"

    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=args.fc_layer_params)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    train_step_counter = tf.Variable(0)

    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    tf_agent.initialize()

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        tf_agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)
    table = reverb.Table(
        table_name,
        max_size=args.replay_buffer_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        table_name=table_name,
        sequence_length=None,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddEpisodeObserver(
        replay_buffer.py_client,
        table_name,
        args.replay_buffer_capacity
    )

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    for _ in range(args.num_iterations):

        # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(
            train_py_env, tf_agent.collect_policy, args.collect_episodes_per_iteration, rb_observer)

        # Use data from the buffer and update the agent's network.
        iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
        trajectories, _ = next(iterator)
        train_loss = tf_agent.train(experience=trajectories)

        replay_buffer.clear()

        step = tf_agent.train_step_counter.numpy()

        if step % args.eval_interval == 0:
            avg_return = execute_tf(eval_env, tf_agent.policy, args.num_eval_episodes, debug=False)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))

    # Save policy and the final return
    policy_saver = PolicySaver(tf_agent.policy)
    policy_saver.save("./saved_models/4bins/continuous/" + file_name + "/policy")

    mean, error = execute_tf(eval_env, tf_agent.policy, 2000, debug=True)
    print('-------------------------------')
    print(f'{file_name}: {mean}, {error}')
    print('-------------------------------')

    with open("results_log.txt", "a") as log_file:
        log_file.write(f"Result of {file_name}: {mean}, {error}\n")
