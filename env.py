import torch


class OfflineEnv(object):
    def __init__(self):
        self.i = 0
        self.state = True  # binary state

    def reset(self, labels):
        self.i = 0
        self.labels = labels
        self.state = True

        return self.state

    def step(self, action):
        reward = torch.zeros_like(action)
        if self.state:
            for j in range(reward.shape[0]):
                if action[j] == self.labels[j, self.i]:
                    reward[j] = 1
        else:
            self.state = False

        self.i += 1

        return self.state, reward
