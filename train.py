import torch, os, pickle, datetime
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.custom_model import MyModel, MyConfig
from torch.utils.data import DataLoader
from utils.dataset import Dataset_F3
from tqdm import tqdm
import tensorboardX
import random


def test():
    self_attention = torch.rand((8, 8))
    cross_attention = torch.rand((8, 8))
    cross_attention = torch.tril(cross_attention)
    masks = torch.ones(cross_attention.size()).detach() * -99999
    masks = torch.triu(masks, diagonal=1)
    # self_attention += masks
    softmax_self_attention = torch.nn.functional.softmax(self_attention, dim=-1)
    self_attention = np.array(self_attention)
    softmax_self_attention = np.array(softmax_self_attention)
    cross_attention = np.array(cross_attention)
    masks = np.array(masks)
    pass


def main(max_len):
    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model_output = 'mymodel'
    if not os.path.exists(model_output):
        os.makedirs(model_output)
    dataset = Dataset_F3('train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    valid_dataset = Dataset_F3('test')
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)
    config = MyConfig()
    test_max_len = 1600
    model = MyModel(config, max_len).cuda()
    epoch = 101
    # initialize the optimizer, I used AdamW here.
    optimizer = AdamW(model.parameters(), lr=8e-7, betas=(0.9, 0.98))
    # learning rate scheduler, I did not warm up the model.
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-8)
    loss_func = torch.nn.MSELoss()
    writer = tensorboardX.SummaryWriter()
    accumulation_steps = 8
    steps = 0
    min_valid_loss = 99999
    for e in range(epoch):
        print(f"epoch: {e}")
        random.seed(e)
        for _, batch in tqdm(enumerate(dataloader)):
            steps += 1
            total_p1_vectors = batch['input'].to(torch.float32).cuda()
            total_p2_vectors = batch['output'].to(torch.float32).cuda()
            time_len = total_p1_vectors.shape[1]
            if time_len > max_len:
                for i in range(time_len // max_len):
                    p1_vectors = total_p1_vectors[:, i*max_len:(i+1)*max_len, :]
                    p2_vectors = total_p2_vectors[:, i*max_len:(i+1)*max_len, :]
                    output = model(p1_vectors=p1_vectors, p2_vectors=p2_vectors)
                    loss1 = loss_func(output[:, :, :100], p2_vectors[:, :, :100])
                    loss2 = loss_func(output[:, :, 100:150], p2_vectors[:, :, 100:150])
                    loss3 = loss_func(output[:, :, 150:153], p2_vectors[:, :, 150:153])
                    loss4 = loss_func(output[:, :, 156:], p2_vectors[:, :, 156:])
                    loss = (5*loss1+3*loss2+loss3+loss4) / (time_len // max_len) / accumulation_steps
                    loss.backward()
                    if (i + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        writer_loss = loss * (time_len // max_len) * accumulation_steps
                        print(f'train_loss: {writer_loss}')
                        writer.add_scalar("loss", writer_loss, steps)
                writer_loss = loss * (time_len // max_len) * accumulation_steps
                optimizer.step()
                optimizer.zero_grad()
                print(f'train_loss: {writer_loss}')
                writer.add_scalar("loss", writer_loss, steps)
            else:
                p1_vectors = total_p1_vectors
                p2_vectors = total_p2_vectors
                output = model(p1_vectors=p1_vectors, p2_vectors=p2_vectors)
                # loss = loss_func(output, total_p2_vectors)
                loss1 = loss_func(output[:, :, :100], p2_vectors[:, :, :100])
                loss2 = loss_func(output[:, :, 100:150], p2_vectors[:, :, 100:150])
                loss3 = loss_func(output[:, :, 150:153], p2_vectors[:, :, 150:153])
                loss4 = loss_func(output[:, :, 156:], p2_vectors[:, :, 156:])
                loss = (5*loss1+3*loss2+loss3+loss4) / accumulation_steps
                loss.backward()
                writer_loss = loss * accumulation_steps
                optimizer.step()
                optimizer.zero_grad()
                print(f'train_loss: {writer_loss}')
                writer.add_scalar("loss", writer_loss, steps)
        scheduler.step()
        if e % 10 == 0:
            with torch.no_grad():
                valid_dataset_lenth = len(valid_dataset)
                valid_loss = 0
                for _, batch in enumerate(valid_dataloader):
                    total_p1_vectors = batch['input'].to(torch.float32).cuda()
                    total_p2_vectors = batch['output'].to(torch.float32).cuda()
                    if total_p1_vectors.shape[1] > test_max_len:
                        total_p1_vectors = total_p1_vectors[:, :test_max_len, :]
                        total_p2_vectors = total_p2_vectors[:, :test_max_len, :]
                    # output = model.generate(total_p1_vectors, total_p2_vectors[:, :1, :])
                    output = model(p1_vectors=total_p1_vectors, p2_vectors=total_p2_vectors)
                    loss1 = loss_func(output[:, :, :100], total_p2_vectors[:, :, :100])
                    loss2 = loss_func(output[:, :, 100:150], total_p2_vectors[:, :, 100:150])
                    loss3 = loss_func(output[:, :, 150:153], total_p2_vectors[:, :, 150:153])
                    loss4 = loss_func(output[:, :, 156:], total_p2_vectors[:, :, 156:])
                    loss = (5 * loss1 + 3 * loss2 + loss3 + loss4)
                    valid_loss += loss / valid_dataset_lenth
                print(f'valid_loss: {valid_loss}')
                writer.add_scalar("valid_loss", valid_loss, steps)
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_output, f"{max_len}_f3_model_{time}"))
    writer.close()


if __name__ == '__main__':
    for max_len in [100]:
        main(max_len)
    # main()