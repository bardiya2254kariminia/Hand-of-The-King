import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Dataset , random_split , TensorDataset
from Networks import Qnetwork
from tqdm import tqdm

dataset_path  = "nashrie_data.pt"
ckpt = torch.load(f=dataset_path , map_location="cpu")
input_data = torch.stack(ckpt["input_data"]).to("cuda")
output_data = ckpt["output_data"].to("cuda")
print(f"{input_data.shape=} , {output_data.shape=}")

if __name__ == "__main__":
    net = Qnetwork().to("cuda")
    mse_loss = MSELoss()
    optimizer = Adam(
        params=net.parameters(),
        lr=0.01,
    )
    DATASET = TensorDataset(input_data , output_data)
    BATCH_SIZE = 2
    EPOCHES = 100
    NUM_WORKERS = 2

    dataloader = DataLoader(
        dataset=DATASET,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True
    )

    for epoch in range(EPOCHES):
        net.train()
        print(f"{epoch= }")
        for idx, (input , target) in tqdm(enumerate(dataloader , 1), total=len(dataloader)):
        # for idx, (input , target) in enumerate(dataloader , 1):
            optimizer.zero_grad()
            predicted = net(input).squeeze(dim=1)
            loss = mse_loss(predicted , target)
            loss.backward()
            optimizer.step()

    # Loading the state_dict
    state_dict = net.state_dict()
    torch.save(state_dict ,"heuristic_model.pt")