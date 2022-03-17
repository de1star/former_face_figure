import torch, os, pickle
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.custom_model import MyModel, MyConfig
from torch.utils.data import DataLoader
from utils.dataset import Dataset_F3
from tqdm import tqdm


def main():
    pass


def test():
    dataset = Dataset_F3('train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    config = MyConfig()
    model = MyModel(config).cuda()
    epoch = 3
    # initialize the optimizer, I used AdamW here.
    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98))
    # learning rate scheduler, I did not warm up the model.
    scheduler = CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-5)
    loss_func = torch.nn.MSELoss()
    max_len = 5000
    accumulation_steps = 4
    for e in range(epoch):
        for _, batch in tqdm(enumerate(dataloader)):
            p1_vectors = batch['input'].to(torch.float32).cuda()
            p2_vectors = batch['output'].to(torch.float32).cuda()
            if p1_vectors.shape[1] > max_len:
                p1_vectors = p1_vectors[:, :max_len, :]
                p2_vectors = p2_vectors[:, :max_len, :]
            output = model(p1_vectors=p1_vectors, p2_vectors=p2_vectors)
            loss = loss_func(output, p2_vectors)
            loss = loss / accumulation_steps
            loss.backward()
            if (_ + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(loss)
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    test()
