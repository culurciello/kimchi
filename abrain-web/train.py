# E. Culurciello
# March 2023
# ABRAIN web version neural network


import random
import argparse

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset

from model import ABrainV1
from abrain_dataset import AbrainDataset
from tqdm import tqdm

title = "train ABRAIN"

def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    arg('--seed', type=int, default=987, help='random seed')
    arg('--device_num', type=int, default=0, help='cuda device')
    arg('--batch_size', type=int, default=4,  help='batch size')
    arg('--num_workers', type=int, default=0,  help='number of workers')
    arg('--epochs', type=int, default=10,  help='number of training epochs')
    arg('--reload', type=str, default='', help='path to reload checkpoint')
    arg('--char', action='store_true', default=False, help='use ASCII char (faster but less general) instead of BERT vocab')
    arg('--max_target_length', default=128, help='max length of text sequence')
    arg('--vit_eye_size', type=int,default=224, help='ViT input eye size')
    arg('--image_resize', type=int,default=512, help='resize webpage image to this size')
    arg('--image_encoder_model', type=str, default='google/vit-base-patch16-224-in21k')
    arg('--text_decode_model', type=str, default='bert-base-uncased')
    arg('--trained_model_path', type=str, default="./trained_model")
    arg('--vocab_file', type=str, default="https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt", help='vocabulary file')
    arg('--vocab_size', type=int, default=30522, help='size of the vocabulary')
    arg('--ascii_num', type=int, default=144, help='size of ascii chars')
    arg('--vit_size', type=int, default=768, help='ViT emb dim')
    arg('--d_size', type=int, default=512, help='transformer emb dim')
    arg('--nheads', type=int, default=8, help='transformer MHA heads')
    arg('--dec_layers', type=int, default=6, help='transformer decoder layers')
    arg('--dropout', type=float, default=0.1, help='transformer dropout')
    args = parser.parse_args(raw_args)
    return args


def train(model, criterion, optimizer, scaler, dataset):
    model.train()
    for epoch in range(args.epochs):
        average_loss = 0
        for (input_seq, target_seq) in dataset:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                input_seq=input_seq.to(device)
                target_seq=target_seq.to(device)
                out_seq = model(input_seq, target_seq)
                loss = criterion(out_seq.squeeze(0), target_seq.squeeze(0))
            
            average_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        print(f'epoch {epoch}, loss {average_loss}')

    # save model
    model_file = 'model.pth'
    print(f'saving model to: {model_file}')
    torch.save(model.state_dict(), model_file)


if __name__ == "__main__":
    args = get_args() # Holds all the input arguments

    # setup
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} compute device")
    print(torch.cuda.get_device_properties(0))

    # random seeds and reproducible results:
    random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # np.set_printoptions(precision=2)
    torch.set_printoptions(profile="full", precision=2)
    
    # dataset:
    dataset = load_dataset('../data/web')
    train_data = AbrainDataset(dataset, args, split='train')
    # valid_data = AbrainDataset(dataset, args, split='validation')
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=1, num_workers=0)
    # valid_dataloader = DataLoader(valid_data, batch_size=1, num_workers=0)

    model = ABrainV1(config=args, device=device).to(device)
    # test model:
    # batch = train_data[0]
    # print("TEST model:", batch[0].shape, batch[1].shape)
    # # loss = model.training_step(batch)
    # # print("TEST LOSS:", loss)
    # out = model(batch[0], batch[1])
    # print("TEST out shape:", out.shape)

    # manual train:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler()
    train(model, criterion, optimizer, scaler, train_dataloader)
    print('finished training')
