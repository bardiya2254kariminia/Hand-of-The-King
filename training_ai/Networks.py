import torch
import torch.nn as nn

class Qnetwork(nn.Module):
    def __init__(self, in_c=56 , out_c=1):
        super(Qnetwork, self).__init__()
        self.arc_list= nn.ModuleList()
        channels = [in_c , 128,64,32,out_c]
        for idx , (in_c, out_c) in enumerate(zip(channels[:-1] , channels[1:]),start=1):
            self.arc_list.add_module(
                f"Linear_{idx}" , nn.Linear(in_c , out_c)
            ) 
            self.arc_list.add_module(
                f"relu_{idx}" , nn.ReLU()
            ) 
        self.arc = nn.Sequential(*self.arc_list)

    def forward(self , x):
        return self.arc(x)