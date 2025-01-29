import torch
import torch.nn as nn

class Qnetwork(nn.Module):
    def __init__(self, in_c=56 , out_c=1):
        super(Qnetwork, self).__init__()
        self.arc = nn.Sequential(
            nn.Linear(in_c , 128),
            nn.ReLU(),
            nn.Linear(128 , 64),
            nn.ReLU(),
            nn.Linear(128 , 32),
            nn.ReLU(),
            nn.Linear(32 , out_c),
        )

    def forward(self , x):
        return self.arc(x)