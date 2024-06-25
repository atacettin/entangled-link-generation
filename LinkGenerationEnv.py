import gym
from gym import spaces
import numpy as np

class LinkGenerationEnvV2(gym.Env):
    def __init__(self, n_links, threshold, decay_rate, actions, remove_hopeless_links=False):
        # Necessary Params
        self.threshold = threshold
        self.n_links = n_links
        self.decay_rate = decay_rate
        self.actions = actions
        self.F_max = np.max(actions[:, 0])
        self.n_bins = int(np.ceil((1 / decay_rate) * np.log((self.F_max - 0.25) / (threshold - 0.25))))
        self.remove_hopeless_links = remove_hopeless_links

        assert self.n_links <= self.n_bins, "Cannot establish n links with <n bins"

        self.observation_space = spaces.MultiDiscrete([self.n_links] * self.n_bins)
        self.action_space = spaces.Discrete(len(actions))

        self._action_to_protocol = []
        self._action_to_bin = []
        for i in range(len(actions)):
            self._action_to_protocol.append(actions[i])
            self._action_to_bin.append(
                int(np.ceil((1 / self.decay_rate) * np.log((actions[i][0] - 0.25) / (self.threshold - 0.25)))))

        self.state = None
        self.current_links = None

    def reset(self, seed=None, options=None):
        # default observation
        super().reset(seed=seed)
        self.state = [0] * self.n_bins
        self.current_links = 0

        return self.state

    def step(self, action):

        # Roll the list by 1
        self.current_links -= self.state[-1]
        self.state = [0] + self.state[0:-1]
        # Set first index to 0

        # Get the protocol from the action
        protocol = self._action_to_protocol[action]
        F, p = protocol[0], protocol[1]

        # Check if action succeeds in generating an entangled state
        # Check if the action succeeds in generating an entangled state
        if self.np_random.binomial(1, p):
            bin = self.n_bins - self._action_to_bin[action]
            self.state[bin] += 1
            self.current_links += 1

        # Check if the desired state is achieved:

        # Remove hopeless links
        if self.remove_hopeless_links:
            self.state.reverse()
            for i in range(len(self.state)):
                if self.state[i] > 0:  # a link exists in this state
                    links_left = self.n_links - sum(self.state[i:])
                    if i >= links_left:
                        break
                    else:
                        self.current_links -= self.state[i]
                        self.state[i] = 0
            self.state.reverse()

        if self.current_links == self.n_links:
            terminated, reward = True, 0
        else:
            terminated, reward = False, -1

        return self.state, reward, terminated, {}

    def render(self, mode="human"):
        pass


class LinkGenerationEnvV2_Continuous(gym.Env):
    def __init__(self, n_links, threshold, decay_rate, tradeoff, actions, remove_hopeless_links=False):
        # Necessary Params
        self.threshold = threshold
        self.n_links = n_links
        self.decay_rate = decay_rate
        self.actions = actions  # [p_min, p_max] that can be chosen
        self.F_max = 1 - tradeoff * actions[0]
        self.tradeoff = tradeoff
        self.n_bins = int(np.ceil((1 / decay_rate) * np.log((self.F_max - 0.25) / (threshold - 0.25))))
        self.remove_hopeless_links = remove_hopeless_links

        print("n_links", self.n_links, "n_bins", self.n_bins)
        assert self.n_links <= self.n_bins, "Cannot establish n links with <n bins"

        self.observation_space = spaces.MultiDiscrete([self.n_links] * self.n_bins)
        self.action_space = spaces.Box(low=actions[0], high=actions[1], shape=(1,))

        self.state = None
        self.current_links = None
        self.time_steps = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # default observation
        self.state = [0] * self.n_bins
        self.current_links = 0
        self.time_steps = 0

        return self.state

    def step(self, action):
        if action < self.actions[0] or action > self.actions[1]:
            raise Exception("Action out of range")

        self.current_links -= self.state[-1]
        self.state = [0] + self.state[0:-1]
        # Set first index to 0

        F, p = 1 - self.tradeoff * action, action

        # Check if action succeeds in generating an entangled state
        # Check if the action succeeds in generating an entangled state
        if np.random.binomial(1, p):
            bin = self.n_bins - int(np.ceil((1 / self.decay_rate) * np.log((F - 0.25) / (self.threshold - 0.25))))
            self.state[bin] += 1
            self.current_links += 1

        self.time_steps += 1

        if self.current_links == self.n_links:
            terminated, reward = True, 0
        else:
            terminated, reward = False, -1

        return self.state, reward, terminated, {}

    def render(self, mode="human"):
        pass

    def bins(self, p):
        F = 1 - (self.tradeoff) * (p)
        return int(np.ceil((1 / self.decay_rate) * np.log((F - 0.25) / (self.threshold - 0.25))))

