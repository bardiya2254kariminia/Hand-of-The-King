import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
from Networks import Qnetwork

ckpt = torch.load(f="" , map_location="cpu")
states_common = ckpt["states_common"]
actions_common = ckpt["actions_common"]
rewards_common = ckpt["rewards_common"]
states = ckpt["states_common"]
actions = ckpt["actions_common"]
rewards = ckpt["rewards_common"]


common_net = Qnetwork(in_c=51 , out_c=51)
companion_net = Qnetwork(in_c=57 , out_c=6)
mse_loss = MSELoss()
optimizer_common = Adam(
    params=common_net.parameters(),
    lr=0.01,
)
optimizer_companion = Adam(
    params=companion_net.parameters(),
    lr=0.01,
)
epoches = 100

for epoch in epoches:
    optimizer.zero_grad()
    
    # Get Q-values for all actions
    q_values = q_network(states)
    
    # Extract the Q-values for the chosen actions
    q_values_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Compute the loss (difference between predicted Q-values and Minimax rewards)
    loss = criterion(q_values_for_actions, rewards)
    
    loss.backward()
    optimizer.step()