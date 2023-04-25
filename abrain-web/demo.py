# E. Culurciello, A. Chang
# March 2023
# ABRAIN web version neural network

# demo

import argparse
from difflib import SequenceMatcher
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer

from model import ABrainV1
from abrain_dataset import AbrainDataset
import re
from PIL import Image, ImageDraw, ImageFont
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class bcolors:
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'

title = "demo ABRAIN"

def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    arg('--seed', type=int, default=987, help='random seed')
    arg('--batch_size', type=int, default=1,  help='batch size')
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

def get_bboxes(ss: str):
    lnum = re.findall(r'\d+', ss)
    boxes = []
    i = 0
    while i < len(lnum):
        box = {'x':int(lnum[i]), 'y':int(lnum[i+1]), 'w':int(lnum[i+2]), 'h':int(lnum[i+3])}
        boxes.append(box)
        i += 4
    return boxes

def infer(model, dataset: AbrainDataset):
    model.eval()
    model.to(device)
    idx = 0 #random.choice(range(len(dataset)))

    image = dataset.dataset[idx]['image']
    labels = dataset.dataset[idx]['label']
    
    input_seq, target_seq = dataset[idx]
    input_seq, target_seq = input_seq.unsqueeze(0), target_seq.unsqueeze(0)
    with torch.no_grad():
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        out = model.generate(input_seq)
    
    out = out.squeeze(0).squeeze(0)
    
    read_string = ''
    target_string = ''
    for c in out:
        read_string += chr(c.argmax())

    for c in target_seq.squeeze(0):
        target_string += chr(c.item())

    score = SequenceMatcher(None, read_string, target_string).ratio()
    print(bcolors.OKBLUE + '\ntest:\n' + bcolors.ENDC + read_string)
    print(bcolors.OKCYAN + 'Ground truth similarity score (0-1):' + bcolors.ENDC, score)

    boxes = get_bboxes(read_string)

    image = dataset.dataset[idx]['image'].convert('RGB')
    draw = ImageDraw.Draw(image)
    for box in boxes:    
        actual_box = [box['x'], box['y'], box['x']+box['w'], box['y']+box['h']]
        draw.rectangle(actual_box, outline='blue')
    image.show()


def demo(model, dataset):
    model.eval()
    model.to(device)
    cnt = 0
    for (input_seq, target_seq) in dataset:
        with torch.no_grad():
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            out = model.generate(input_seq, target_seq[0][0:5].unsqueeze(0))
        
        out = out.squeeze(0).squeeze(0)

        read_string = ''
        target_string = ''
        for c in out:
            read_string += chr(c.argmax())

        for c in target_seq.squeeze(0):
            target_string += chr(c.item())

        score = SequenceMatcher(None, read_string, target_string).ratio()
        print(bcolors.OKBLUE + '\ntest:\n' + bcolors.ENDC + read_string)
        print(bcolors.OKCYAN + 'Ground truth similarity score (0-1):' + bcolors.ENDC, score)

        if cnt == 10:
            break
        cnt += 1


def main(args):
    # training configuration:
    vocab_file = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
    config = {
        "im_size": 224, # ViT input eye size
        "resize": 512, # resize webpage image to this size
        "max_epochs": 100,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "lr": 5e-5,
        "train_batch_sizes": [1],
        "val_batch_sizes": [1],
        "verbose": False,
        "vocab_file": vocab_file, # vocabulary
        "vocab_size": 30522, # vocabulary size
        "ascii_num": 144, # number of ASCII characters
        "char": True if args.char else False
    }

    model = ABrainV1(config=config, device=device)
    model.load_state_dict(torch.load('model.pth')) # load trained model weights
    # if args.reload:
    #     checkpoint = torch.load(args.reload)
    #     model.load_state_dict(checkpoint["state_dict"])

    # dataset:
    dataset = load_dataset('./data/')
    # dataset = load_dataset('achang/form_boxes')
    valid_data = AbrainDataset(dataset, config, split='train')
    valid_dataloader = DataLoader(valid_data, batch_size=1, num_workers=0)
    
    # run demo:
    # infer(model, valid_data)
    demo(model, valid_dataloader)


if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ABrainV1(config=args, device=device)
    model.load_state_dict(torch.load('model.pth')) # load trained model weights
    model.to(device)
    model.eval()

    # dataset:
    dataset = load_dataset('../data/web')
    valid_data = AbrainDataset(dataset, args, split='validation')
    valid_dataloader = DataLoader(valid_data, batch_size=1, num_workers=0)
    tokenizer = valid_data.tokenizer
    
    # run demo:
    for (input_seq, target_seq) in valid_dataloader:
        input_seq=input_seq.to(device)
        target_seq=target_seq.to(device)
        with torch.no_grad():
            output_seq = model.generate(input_seq, target_seq[:,0:10], max_length=target_seq.shape[1])
        
        output_seq = output_seq.squeeze(0).squeeze(0)
        
        pred_string = ''
        target_string = ''

        if args.char:
            for c in out:
                pred_string += chr(c.item())
                target_string += chr(c.item())
        else:
            pred_string = tokenizer.decode(output_seq)
            target_string = tokenizer.decode(target_seq[0])

        score = SequenceMatcher(None, pred_string, target_string).ratio()
        print(bcolors.OKBLUE + '\ntest:\n' + bcolors.ENDC + pred_string)
        print(bcolors.OKCYAN + 'Ground truth similarity score (0-1):' + bcolors.ENDC, score)
