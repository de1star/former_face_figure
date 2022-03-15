import torch, os, pickle
from transformers import LongformerTokenizer, LongformerConfig, LongformerForQuestionAnswering
from torch.utils.data import DataLoader
from utils.dataset import Dataset_F3


def main():
    config = LongformerConfig()
    model = LongformerForQuestionAnswering(config)
    dataset = Dataset_F3('valid')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    epoch = 1
    for e in range(epoch):
        for _, batch in enumerate(dataloader):
            input = batch['input'].squeeze(0)[5:]
            true_output = batch['output'].squeeze(0)[5:]
            attention_mask = torch.ones(input.shape[0])
            output = model(attention_mask=attention_mask, inputs_embeds=input)
            print(output)


if __name__ == '__main__':
    main()
