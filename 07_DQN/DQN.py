import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

class Q_net(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Q_net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1)
        return self.fc2(x)
    

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen = capacity)

    def add(self, state, action, reward, next_state, done):  
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self):
        return len(self.buffer)


class DQN:

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, 
                target_update, device):
        self.action_dim = action_dim
        self.q_net = Q_net(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = Q_net(state_dim, hidden_dim, self.action_dim).to(device)
        




