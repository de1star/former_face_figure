import torch, os, pickle, sys, argparse
from torch.utils.data import DataLoader
from utils.dataset import Dataset_F3
from utils.custom_model import MyModel, MyConfig


def main(args):
    config = MyConfig()
    model = MyModel(config, 100)
    model_path = args.path
    print(f'load model from {model_path}')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    valid_dataset = Dataset_F3('test')
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(valid_dataloader):
            print(_)
            path = f'demos/tensors/{_}'
            if not os.path.exists(path):
                os.makedirs(path)
            total_p1_vectors = batch['input'].to(torch.float32).cpu()
            total_p2_vectors = batch['output'].to(torch.float32).cpu()
            total_p1_vectors = total_p1_vectors[:, :5000, :]
            total_p2_vectors = total_p2_vectors[:, :5000, :]
            # output = model.generate(total_p1_vectors, total_p2_vectors[:, :1, :])
            output = model(p1_vectors=total_p1_vectors, p2_vectors=total_p2_vectors)
            torch.save(total_p1_vectors, f'{path}/input.pt')
            torch.save(total_p2_vectors, f'{path}/true_output.pt')
            torch.save(output, f'{path}/my_output.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', help='model path')
    args = parser.parse_args()
    main(args)
