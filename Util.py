import numpy as np


def pick_actions_uniformly(lam, hi, lo, n):
    width = (hi - lo) / (n - 1)
    acts = []
    for i in range(n):
        p = lo + i * width
        acts.append([1 - lam * p, p])
    return acts


def bins(p, tradeoff, decay_rate, threshold):
    F = 1 - (tradeoff) * (p)
    return int(np.ceil((1 / decay_rate) * np.log((F - 0.25) / (threshold - 0.25))))


def bin_probability(bin, tradeoff, decay_rate, threshold, actions):
    return min((1 / tradeoff) * (0.75 - (threshold - 0.25) * np.exp(bin * decay_rate)) - 0.000001, actions[1])


def F(p, tradeoff):
    return 1 - tradeoff * p


def execute(env, policy, n=10):
    returns = []
    for i in range(n):
        terminated = False
        state = env.reset()
        episode_return = 0
        while not terminated:
            action = policy.action(state)
            state, reward, terminated, info = env.step(action)
            episode_return += reward
            # print(state)
        episode_return -= 1
        returns.append(episode_return)
    return returns


def execute_tf(env, policy, n=10, debug=False):
    returns = []
    for i in range(n):
        if debug and i % 100 == 0:
            print("Step:", i)
        time_step = env.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action = policy.action(time_step)
            time_step = env.step(action.action)
            episode_return += time_step.reward

        episode_return -= 1
        returns.append(episode_return)
    return np.mean(returns), 3*np.std(returns)/np.sqrt(n)


def collect_episode(env, heuristic):
    actions = []
    state = env.reset()
    terminated = False
    while not terminated:
        action = heuristic.action(state)
        state, reward, terminated, info = env.step(action)
        actions.append((action, state))
    return actions
