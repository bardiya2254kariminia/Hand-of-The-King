# from utils.training_utils import *
# from training_ai.Networks import *



# model = Qnetwork(2,2)
# for name , param in model.named_parameters():
#     print(param.data)
#     param.data = torch.full_like(param.data , 0)
#     break
# for name , param in model.named_parameters():
#     print(param.data)
#     break
import torch

ckpt = torch.load(r"C:\Users\Asus\Desktop\SBU\Sem 5 SBU\Artificail Inteligence(Classic)\Final_Project\Hand-of-The-King\nashrie_data.pt",map_location="cpu")
a = ckpt["output_data"]
print(a.shape)
