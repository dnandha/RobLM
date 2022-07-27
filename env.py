import torch


class OfflineEnv(object):
    def __init__(self):
        self.i = 0
        self.i_total = 0
        self.done = False
        self.reward_mu = None

    def reset(self, labels):
        self.i = 0
        self.done = False
        self.labels = labels

        if self.reward_mu is None:
            self.reward_mu = torch.zeros(labels.shape[0], dtype=float, device=self.labels.get_device())

    def step(self, action, R=1, gamma=0.99):
        reward = torch.zeros(self.labels.shape[0], dtype=float, device=self.labels.get_device())
        for j in range(reward.shape[0]):
            # once done, no reward is given
            if not self.done:
                if action[j] == self.labels[j, self.i]:
                    # subtract running mean to fight high variance  # TODO
                    reward[j] = gamma**self.i * R #(R - self.reward_mu)
            else:
                reward[j] = 0

        # scale reward by episode length
        #reward /= self.labels.shape[1]

        # update running mean
        if self.i_total > 0:
            self.reward_mu = ((self.i_total - 1) * self.reward_mu + reward) / self.i_total

        # we're done when no reward received at any point is 0
        self.done = reward == 0

        # step simulator
        self.i += 1
        self.i_total += 1

        return self.done, reward
