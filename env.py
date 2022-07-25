import torch


class OfflineEnv(object):
    def __init__(self):
        self.i = 0
        self.done = False
        self.reward = None

    def reset(self, labels):
        self.i = 0
        self.done = False
        self.labels = labels
        self.reward = torch.zeros_like(labels)

    def step(self, action, R=1, gamma=0.99):
        if not self.done:
            for j in range(self.reward.shape[0]):
                if action[j] == self.labels[j, self.i]:
                    self.reward[j] = gamma**self.i * R

        reward = torch.sum(self.reward, dim=1)
        self.done = reward == 0
        self.i += 1

        return self.done, reward
