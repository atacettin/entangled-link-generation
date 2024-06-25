from typing import List
import numpy as np


class SAHeuristic:
    def __init__(self, p_in):
        self.p_in = p_in

    def action(self, state):
        return self.p_in


class MatchingHeuristic:
    def __init__(self, p_in, n_links, threshold, decay_rate, tradeoff, actions):
        self.p_in = p_in
        self.n_links = n_links
        self.threshold = threshold
        self.decay_rate = decay_rate
        self.tradeoff = tradeoff
        self.actions = actions
        self.F_max = 1 - tradeoff * actions[0]

    def bins(self, p):
        F = 1 - (self.tradeoff) * (p)
        return int(np.ceil((1 / self.decay_rate) * np.log((F - 0.25) / (self.threshold - 0.25))))

    def action(self, state: List):
        stcopy = state.copy()
        # links_left = self.n_links - sum(stcopy)

        # Finding the MV link
        stcopy.reverse()
        mv = -1
        for i in range(len(stcopy)):
            if stcopy[i] > 0:  # a link exists in this state
                links_left = self.n_links - sum(stcopy[i:])
                if i >= links_left:
                    if links_left == 1:
                        return self.actions[1] - 0.000001
                    mv = i - 1
                    break

        if mv == -1:
            return self.p_in
        else:
            return min((1 / self.tradeoff) * (0.75 - (self.threshold - 0.25) * np.exp(mv * self.decay_rate)) - 0.000001,
                       self.actions[1])


class PolicyHeuristic:
    def __init__(self, alpha, n_links, threshold, decay_rate, tradeoff, actions):
        self.alpha = alpha
        self.n_links = n_links
        self.threshold = threshold
        self.decay_rate = decay_rate
        self.tradeoff = tradeoff
        self.actions = actions
        self.F_max = 1 - tradeoff * actions[0]

    def action(self, state: List):
        stcopy = state.copy()
        stcopy.reverse()
        a = -1
        for i in range(len(stcopy)):
            if i >= self.n_links - sum(stcopy[i + 1:]) - 1:
                a = self.alpha * i
                break
        return min((1 / self.tradeoff) * (0.75 - (self.threshold - 0.25) * np.exp(a * self.decay_rate)),
                   self.actions[1]) - 0.001


class RandomHeuristic:
    def __init__(self, actions):
        self.actions = actions

    def action(self, state: List):
        return np.random.uniform(low=self.actions[0], high=self.actions[1])

