import torch
import torch.nn as nn
import numpy as np
import random
import os
import torch.optim as opt

from RL.utils import utils
from RL.models.dqn import MolDQN
from rdkit import Chem
from rdkit.Chem import QED
from RL.environment.environment import Molecule
from collections import deque

from RL.config import config

class Agent(): 
    def __init__(self, input_length, output_length, device) -> None:
        self.device = device
        self.dqn, self.target_dqn = (
            MolDQN(input_length, output_length).to(self.device),
            MolDQN(input_length, output_length).to(self.device)
        )
        self.replay_buffer = deque(maxlen=config.REPLAY_BUFFER_CAPACITY)

        for p in self.target_dqn.parameters(): 
            p.requires_grad = False

        self.optimizer = opt.Adam(self.dqn.parameters(), lr=config.LEARNING_RATE)
   
    def load_model_parameters(self, dqn_path, model): 
        try: 
            self.dqn.load_state_dict(torch.load(os.path.join(dqn_path, model)))
            print("Loaded successfully!")
        except: 
            print("Faild loading from path")

    def get_action(self, observations, epsilon_threshold): 
        
        if np.random.uniform() < epsilon_threshold: 
            action = np.random.randint(0, observations.shape[0])
        else:
            q_value = self.dqn.forward(observations.to(self.device)).cpu()
            action = torch.argmax(q_value).numpy()


        return action


    def update_parameters(self, batch_size, gamma, polyak):
        
        mini_sample = random.sample(self.replay_buffer, batch_size)
        states, rewards, next_states, dones = zip(*mini_sample)

        q_t = torch.zeros(batch_size, 1, requires_grad=False)
        v_tp1 = torch.zeros(batch_size, 1, requires_grad=False)


        states = (
                torch.FloatTensor(states)
                .reshape(-1, config.FINGERPRINT_LENGTH + 1)
                .to(self.device)
            )

        q_t = self.dqn(states)

        for i in range(config.BATCH_SIZE):
            next_state = (
                    torch.FloatTensor(next_states[i])
                    .reshape(-1, config.FINGERPRINT_LENGTH + 1)
                    .to(self.device)
                )
            
            v_tp1[i] = torch.max(self.target_dqn(next_state))

      
        rewards = torch.FloatTensor(rewards).reshape(q_t.shape).to(self.device)
        q_t = q_t.to(self.device)
        v_tp1 = v_tp1.to(self.device)
        dones = torch.FloatTensor(dones).reshape(q_t.shape).to(self.device)
    
        # # get q values
        q_tp1_masked = (1 - dones) * v_tp1 # is terminal sate, no future reward
        q_t_target = rewards + gamma * q_tp1_masked # rewards here already discounted, why multiply target with gamma?
        td_error = q_t - q_t_target

        # Huber Loss
        q_loss = torch.where(
        torch.abs(td_error) < 1.0,
        0.5 * td_error * td_error,
        1.0 * (torch.abs(td_error) - 0.5),
        )
        q_loss = q_loss.mean()

           # backpropagate
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        # polyak averaging 
        with torch.no_grad():
            for p, p_targ in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
    
        return q_loss
    



        
