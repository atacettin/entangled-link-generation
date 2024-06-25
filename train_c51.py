import argparse
import ast
import os
import tensorflow as tf
import gym
import numpy as np
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy, PolicySaver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from Util import execute_tf
from LinkGenerationEnv import LinkGenerationEnvV2

import warnings

warnings.filterwarnings("ignore")


def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_links", default=4, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--decay_rate", default=0.2, type=float)
    parser.add_argument("--actions", default=[[0.72, 0.4]])
    parser.add_argument("--num_iterations", default=20000, type=int)
    parser.add_argument("--initial_collect_steps", default=2000, type=int)
    parser.add_argument("--replay_buffer_capacity", default=2000, type=int)
    parser.add_argument("--fc_layer_params", default=(256, 256))
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--gamma", default=1, type=float)
    parser.add_argument("--eval_interval", default=5000, type=int)
    parser.add_argument("--num_eval_episodes", default=50, type=int)
    parser.add_argument("--num_atoms", default=81, type=int)
    parser.add_argument("--min_q_value", default=-500, type=int)
    parser.add_argument("--max_q_value", default=0, type=int)
    args = parser.parse_args()

    actions = ast.literal_eval(args.actions)

    env_name = "LinkGenerationEnv"
    file_name = f"{args.n_links}links_{len(actions)}actions"
    print("---------------------------------------")
    print(f"Links:{args.n_links}, Actions:{actions}")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    gym.envs.register(
        id='LinkGenerationEnv',
        entry_point='__main__:LinkGenerationEnvV2',
        max_episode_steps=500,
        kwargs={'n_links': args.n_links, 'threshold': args.threshold, 'decay_rate': args.decay_rate,
                'actions': np.array(actions),
                'remove_hopeless_links': False}
    )

    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    categorical_q_net = categorical_q_network.CategoricalQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        num_atoms=args.num_atoms,
        fc_layer_params=args.fc_layer_params
    )

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)

    train_step_counter = tf.Variable(0)

    agent = categorical_dqn_agent.CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        categorical_q_network=categorical_q_net,
        optimizer=optimizer,
        min_q_value=args.min_q_value,
        max_q_value=args.max_q_value,
        n_step_update=2,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=args.gamma,
        train_step_counter=train_step_counter)
    agent.initialize()

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=args.replay_buffer_capacity)

    for _ in range(args.initial_collect_steps):
        collect_step(train_env, random_policy)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=args.batch_size,
        num_steps=3).prefetch(3)

    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    for _ in range(args.num_iterations):
        collect_step(train_env, agent.collect_policy)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience)

        step = agent.train_step_counter.numpy()

        if step % args.eval_interval == 0:
            avg_return = execute_tf(eval_env, agent.policy, args.num_eval_episodes, debug=False)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))

    # Save policy and the final return
    policy_saver = PolicySaver(agent.policy)
    policy_saver.save("./saved_models/4bins/discrete/" + file_name + "/policy")

    mean, error = execute_tf(eval_env, agent.policy, 2000, debug=True)

    print('-------------------------------')
    print(f'{file_name}: {mean}, {error}')
    print('-------------------------------')

    with open("results_log.txt", "a") as log_file:
        log_file.write(f"Result of {file_name}: {mean}, {error}\n")
