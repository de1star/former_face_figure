import torch, os, pickle, datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.custom_model import MyModel, MyConfig
from torch.utils.data import DataLoader
from utils.dataset import Dataset_F3
from tqdm import tqdm
import tensorboardX
import random


def main():
    pass


def test():
    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model_output = 'mymodel'
    if not os.path.exists(model_output):
        os.makedirs(model_output)
    dataset = Dataset_F3('train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    valid_dataset = Dataset_F3('test')
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)
    config = MyConfig()
    model = MyModel(config).cuda()
    epoch = 100
    # initialize the optimizer, I used AdamW here.
    optimizer = AdamW(model.parameters(), lr=2e-6, betas=(0.9, 0.98))
    # learning rate scheduler, I did not warm up the model.
    scheduler = CosineAnnealingLR(optimizer, T_max=99, eta_min=1e-8)
    loss_func = torch.nn.MSELoss()
    writer = tensorboardX.SummaryWriter()
    accumulation_steps = 8
    steps = 0
    max_len = 500
    test_max_len = 800
    for e in range(epoch):
        random.seed(e)
        for _, batch in tqdm(enumerate(dataloader)):
            steps += 1
            p1_vectors = batch['input'].to(torch.float32).cuda()
            p2_vectors = batch['output'].to(torch.float32).cuda()
            if p1_vectors.shape[1] > max_len:
                start = random.randint(0, p1_vectors.shape[1] - max_len)
                p1_vectors = p1_vectors[:, start:start+max_len, :]
                p2_vectors = p2_vectors[:, start:start+max_len, :]
            output = model(p1_vectors=p1_vectors, p2_vectors=p2_vectors)
            # loss = loss_func(output, p2_vectors)
            loss1 = loss_func(output[:, :, :100], p2_vectors[:, :, :100])
            loss2 = loss_func(output[:, :, 100:150], p2_vectors[:, :, 100:150])
            loss3 = loss_func(output[:, :, 150:153], p2_vectors[:, :, 150:153])
            loss4 = loss_func(output[:, :, 156:], p2_vectors[:, :, 156:])
            loss = (5*loss1+3*loss2+loss3+loss4) / accumulation_steps
            loss.backward()
            if (_ + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(f'train_loss: {loss * accumulation_steps}')
                writer.add_scalar("loss", loss * accumulation_steps, steps)
        optimizer.step()
        optimizer.zero_grad()
        print(f'train_loss: {loss * accumulation_steps}')
        writer.add_scalar("loss", loss * accumulation_steps, steps)
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(model_output, f"f3_model_{time}"))
        if (e+1) % 10 == 0:
            valid_dataset_lenth = len(valid_dataset)
            valid_loss = 0
            for _, batch in enumerate(valid_dataloader):
                p1_vectors = batch['input'].to(torch.float32).cuda()
                p2_vectors = batch['output'].to(torch.float32).cuda()
                if p1_vectors.shape[1] > test_max_len:
                    p1_vectors = p1_vectors[:, :test_max_len, :]
                    p2_vectors = p2_vectors[:, :test_max_len, :]
                output = model.generate(p1_vectors)
                loss1 = loss_func(output[:, :, :100], p2_vectors[:, :, :100])
                loss2 = loss_func(output[:, :, 100:150], p2_vectors[:, :, 100:150])
                loss3 = loss_func(output[:, :, 150:153], p2_vectors[:, :, 150:153])
                loss4 = loss_func(output[:, :, 156:], p2_vectors[:, :, 156:])
                loss = (5 * loss1 + 3 * loss2 + loss3 + loss4)
                valid_loss += loss / valid_dataset_lenth
            print(f'valid_loss: {valid_loss}')
            writer.add_scalar("valid_loss", valid_loss, steps)
    writer.close()


if __name__ == '__main__':
    test()
