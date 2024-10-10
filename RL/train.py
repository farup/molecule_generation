import torch
import sys 
sys.path.append(r"C:\Users\tnf\Documents\NTNU\2024\WaterlooInternship\final_project")


from agent import Agent
from environment.environment import QEDRewardMolecule
from config import config

import os
import math
from utils import utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

environment = QEDRewardMolecule(
    discount_factor=config.DISCOUNT_FACTOR,
    atom_types=set(config.ATOM_TYPES),
    init_mol=config.INIT_MOL,
    allow_removal=config.ALLOW_REMOVAL,
    allow_no_modification=config.ALLOW_NO_MODFICATION,
    allow_bonds_between_rings=config.ALLOW_BOND_BETWEEN_RINGS,
    allowed_ring_sizes=set(config.ALLOWED_RING_SIZES),
    max_steps=config.MAX_STEPS,
)

agent = Agent(config.FINGERPRINT_LENGTH + 1, 1, device)

if config.TENSORBOARD_LOG:
    writer = SummaryWriter()

environment.initialize()
eps_threshold = 1.0
batch_losses = []

epsiodes = 0

for it in range(config.ITERATIONS):

    steps_left = config.MAX_STEPS - environment.num_steps_taken

    # Compute a list of all possible valid actions. (Here valid_actions stores the states after taking the possible actions)
    valid_actions = list(environment.get_valid_actions())

    # Append each valid action to steps_left and store in observations.
    observations = np.vstack(
        [
            np.append(
                utils.get_fingerprint(
                    act, config.FINGERPRINT_LENGTH, config.FINGERPRINT_RADIUS
                    ),steps_left,) 
                for act in valid_actions
        ]
    )  # (num_actions, fingerprint_length)

    observations_tensor = torch.Tensor(observations)
    # Get action through epsilon-greedy policy with the following scheduler.
    # eps_threshold = hyp.epsilon_end + (hyp.epsilon_start - hyp.epsilon_end) * \
    #     math.exp(-1. * it / hyp.epsilon_decay)

    a = agent.get_action(observations_tensor, eps_threshold)

    # Find out the new state (we store the new state in "action" here. Bit confusing but taken from original implementation)
    action = valid_actions[a]
    # Take a step based on the action
    result = environment.step(action)

    action_fingerprint = np.append(
        utils.get_fingerprint(action, config.FINGERPRINT_LENGTH, config.FINGERPRINT_RADIUS),
        steps_left,
    )

    next_state, reward, done = result

    # Compute number of steps left
    steps_left = config.MAX_STEPS - environment.num_steps_taken

    # Append steps_left to the new state and store in next_state
    next_state = utils.get_fingerprint(
        next_state, config.FINGERPRINT_LENGTH, config.FINGERPRINT_RADIUS
    )  # (fingerprint_length)

    action_fingerprints = np.vstack(
        [
            np.append(
                utils.get_fingerprint(
                    act, config.FINGERPRINT_LENGTH, config.FINGERPRINT_RADIUS
                ),
                steps_left,
            )
            for act in environment.get_valid_actions()
        ]
    )  # (num_actions, fingerprint_length + 1)

    # Update replay buffer (state: (fingerprint_length + 1), action: _, reward: (), next_state: (num_actions, fingerprint_length + 1),
    # done: ()

    agent.replay_buffer.append((
        action_fingerprint,  # (fingerprint_length + 1)
        reward,
        action_fingerprints,  # (num_actions, fingerprint_length + 1)
        float(result.terminated))
    )

    if done:
        final_reward = reward
        if epsiodes != 0 and config.TENSORBOARD_LOG and len(batch_losses) != 0:
            writer.add_scalar("episode_reward", final_reward, epsiodes)
            writer.add_scalar("episode_loss", np.array(batch_losses).mean(), epsiodes)
        if epsiodes != 0 and epsiodes % 2 == 0 and len(batch_losses) != 0:
            print(
                "reward of final molecule at episode {} is {}".format(
                    epsiodes, final_reward
                )
            )
            print(
                "mean loss in episode {} is {}".format(
                    epsiodes, np.array(batch_losses).mean()
                )
            )

        epsiodes += 1
        eps_threshold *= 0.99907
        batch_losses = []
        environment.initialize()

    if it % config.UPDATE_INTERVAL == 0 and agent.replay_buffer.__len__() >= config.BATCH_SIZE:
        for update in range(config.NUM_UPDATES_PER_IT):
            loss = agent.update_parameters(config.BATCH_SIZE, config.GAMMA, config.POLYAK)
            loss = loss.item()
            batch_losses.append(loss)

    if  it != 0 and it % config.SAVE_MODEL_EVERY_IT == 0:
        torch.save(agent.dqn.state_dict(), os.path.join(config.SAVE_MODEL_PATH, f"dqn_it_{it}.pt"))
        torch.save(agent.target_dqn.state_dict(), os.path.join(config.SAVE_MODEL_PATH, f"dqn_target_it_{it}.pt"))
