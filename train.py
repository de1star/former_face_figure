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


def main():
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


def test(max_len):
    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model_output = 'mymodel'
    if not os.path.exists(model_output):
        os.makedirs(model_output)
    dataset = Dataset_F3('train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    valid_dataset = Dataset_F3('test')
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)
    config = MyConfig()
    test_max_len = max_len
    training_data_length = 1600
    model = MyModel(config, max_len).cuda()
    epoch = 101
    # initialize the optimizer, I used AdamW here.
    optimizer = AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.98))
    # learning rate scheduler, I did not warm up the model.
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-8)
    loss_func = torch.nn.MSELoss()
    writer = tensorboardX.SummaryWriter()
    accumulation_steps = 8
    steps = 0

    memory_usage = 0
    for e in range(epoch):
        print(f"epoch: {e}")
        random.seed(e)
        for _, batch in tqdm(enumerate(dataloader)):
            steps += 1
            p1_vectors = batch['input'].to(torch.float32).cuda()
            p2_vectors = batch['output'].to(torch.float32).cuda()
            if p1_vectors.shape[1] >= training_data_length:
                start = random.randint(0, p1_vectors.shape[1] - training_data_length)
                long_p1_vectors = p1_vectors[:, start:start+training_data_length, :]
                long_p2_vectors = p2_vectors[:, start:start+training_data_length, :]
                for i in range(training_data_length // max_len):
                    p1_vectors = long_p1_vectors[:, i*max_len:(i+1)*max_len, :]
                    p2_vectors = long_p2_vectors[:, i*max_len:(i+1)*max_len, :]
                    output = model(p1_vectors=p1_vectors, p2_vectors=p2_vectors)
                    loss1 = loss_func(output[:, :, :100], p2_vectors[:, :, :100])
                    loss2 = loss_func(output[:, :, 100:150], p2_vectors[:, :, 100:150])
                    loss3 = loss_func(output[:, :, 150:153], p2_vectors[:, :, 150:153])
                    loss4 = loss_func(output[:, :, 156:], p2_vectors[:, :, 156:])
                    loss = (5*loss1+3*loss2+loss3+loss4) / (training_data_length // max_len) / accumulation_steps
                    memory_usage = max(memory_usage, torch.cuda.memory_allocated())
                    loss.backward()
                writer_loss = loss * (training_data_length // max_len) * accumulation_steps
            else:
                start = random.randint(0, p1_vectors.shape[1] - max_len)
                long_p1_vectors = p1_vectors[:, start:start + max_len, :]
                long_p2_vectors = p2_vectors[:, start:start + max_len, :]
                output = model(p1_vectors=p1_vectors, p2_vectors=p2_vectors)
                # loss = loss_func(output, p2_vectors)
                loss1 = loss_func(output[:, :, :100], p2_vectors[:, :, :100])
                loss2 = loss_func(output[:, :, 100:150], p2_vectors[:, :, 100:150])
                loss3 = loss_func(output[:, :, 150:153], p2_vectors[:, :, 150:153])
                loss4 = loss_func(output[:, :, 156:], p2_vectors[:, :, 156:])
                loss = (5*loss1+3*loss2+loss3+loss4) / accumulation_steps
                memory_usage = max(memory_usage, torch.cuda.memory_allocated())
                loss.backward()
                writer_loss = loss * accumulation_steps
            if (_ + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(f'train_loss: {writer_loss}')
                writer.add_scalar("loss", writer_loss, steps)
        optimizer.step()
        optimizer.zero_grad()
        print(f'train_loss: {loss * accumulation_steps}')
        writer.add_scalar("loss", loss * accumulation_steps, steps)
        scheduler.step()
        if e % 10 == 0:
            with torch.no_grad():
                valid_dataset_lenth = len(valid_dataset)
                valid_loss = 0
                for _, batch in enumerate(valid_dataloader):
                    p1_vectors = batch['input'].to(torch.float32).cuda()
                    p2_vectors = batch['output'].to(torch.float32).cuda()
                    if p1_vectors.shape[1] > test_max_len:
                        p1_vectors = p1_vectors[:, :test_max_len, :]
                        p2_vectors = p2_vectors[:, :test_max_len, :]
                    # output = model.generate(p1_vectors, p2_vectors[:, :1, :])
                    output = model(p1_vectors=p1_vectors, p2_vectors=p2_vectors)
                    loss1 = loss_func(output[:, :, :100], p2_vectors[:, :, :100])
                    loss2 = loss_func(output[:, :, 100:150], p2_vectors[:, :, 100:150])
                    loss3 = loss_func(output[:, :, 150:153], p2_vectors[:, :, 150:153])
                    loss4 = loss_func(output[:, :, 156:], p2_vectors[:, :, 156:])
                    loss = (5 * loss1 + 3 * loss2 + loss3 + loss4)
                    valid_loss += loss / valid_dataset_lenth
                print(f'valid_loss: {valid_loss}')
                writer.add_scalar("valid_loss", valid_loss, steps)
    torch.save(model.state_dict(), os.path.join(model_output, f"f3_model_{time}"))
    writer.add_scalar("memory_usage", memory_usage / 1.07e9, model.max_len)
    writer.close()


if __name__ == '__main__':
    for max_len in [800, 500, 300, 100, 50]:
        test(max_len)
    # main()