import argparse
from collections import namedtuple
from itertools import count
import pickle
import time

import os, random
import numpy as np
from collections import deque

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter

from sac_agent import SAC
from replay_memory import ReplayMemory

from tensorboardX import SummaryWriter


'''
Implementation of soft actor critic, dual Q network version 
Original paper: https://arxiv.org/abs/1801.01290
Not the author's implementation !
'''

gamma=0.99
batch_size=256
lr=5e-4
hidden_size=400
tau=0.005
alpha=0.2
start_steps=10000
update_start_steps=1e4
reward_scale = 5



device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()

def init_parser():

    parser.add_argument("--env_name", default="BipedalWalkerHardcore-v3")  # OpenAI gym environment name

    parser.add_argument('--capacity', default=2000000, type=int) # replay buffer size
    parser.add_argument('--iteration', default=100000, type=int) #  num of  games
    parser.add_argument('--batch_size', default=256, type=int) # mini batch size
    parser.add_argument('--seed', default=10, type=int)

    # optional parameters
    parser.add_argument('--render', default=False, type=bool) # show UI or not
    parser.add_argument('--train', default=False, type=bool) 
    parser.add_argument('--eval', default=True, type=bool) 
    parser.add_argument('--load', default=False, type=bool) # load model
    parser.add_argument('--render_interval', default=0, type=int) # after render_interval, the env.render() will work

init_parser()
args = parser.parse_args()

# class NormalizedActions(gym.ActionWrapper):
#     def action(self, action):
#         low = self.action_space.low
#         high = self.action_space.high

#         action = low + (action + 1.0) * 0.5 * (high - low)
#         action = np.clip(action, low, high)

#         return action

#     def reverse_action(self, action):
#         low = self.action_space.low
#         high = self.action_space.high

#         action = 2 * (action - low) / (high - low) - 1
#         action = np.clip(action, low, high)

#         return action


env = gym.make(args.env_name)

# Set seeds
env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

writer = SummaryWriter('./logs_data')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]



def main():
    agent = SAC(state_dim, env.action_space, device, hidden_size, lr, gamma, tau, alpha)
    replay_buffer = ReplayMemory(args.capacity, args.seed)

    if args.train: print("Train True")
    if args.load: 
        print("Load True")
        # agent.load_model(actor_path="./models_hard1/actor.pth", critic_path="./models_hard1/critic.pth")
        agent.load_model()
    
    updates = 0

    total_steps = 0
    time_start = time.time()
    scores_deque = deque(maxlen=100)
    avg_scores_array = []

    for i in range(args.iteration):
        ep_r = 0
        ep_s = 0
        done = False
        state = env.reset()
        error_count = 0
        while not done:
            action = []
            if total_steps < start_steps and not args.load:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, True)

            

            next_state, reward, done, info = env.step(action)

            if reward < -0.009:
                error_count += 1
                if error_count == 80:
                    reward = -5.0


            reward = reward * reward_scale
            
            ep_r += reward
            ep_s += 1
            total_steps += 1
            if args.render and i >= args.render_interval : env.render()
            
            mask = 1 if (ep_s == 2000 or error_count == 80) else float(not done)
            if args.train:
                replay_buffer.push(state, action, reward, next_state, mask)

            if error_count == 100:
                break
            
            

            state = next_state
        if args.train:
            for upi in range(ep_s):
                if args.load:
                    if len(replay_buffer) >= 100000:
                        agent.update_parameters(replay_buffer, batch_size, updates)
                        updates += 1
                if not args.load and len(replay_buffer) >= update_start_steps:
                    agent.update_parameters(replay_buffer, batch_size, updates)
                    updates += 1


        scores_deque.append(ep_r)
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        writer.add_scalar('reward/train', ep_r, i)
        writer.add_scalar('reward/avg', avg_score, i)
        s =  (int)(time.time() - time_start)
        print("Ep.: {}, Total Steps: {}, Ep.Steps: {}, Score: {:.2f}, Avg.Score: {:.2f}, Time: {:02}:{:02}:{:02}".\
            format(i, total_steps, ep_s, ep_r, avg_score, \
                  s//3600, s%3600//60, s%60))


        if args.train:
            if ep_r > 1400:
                agent.save_model()

            if i % 20 == 0:
                agent.save_model()




    
    env.close()


if __name__ == '__main__':
    main()