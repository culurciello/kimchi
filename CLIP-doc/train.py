# E. Culurciello, September 2022
# CLIP DOC 

# inspired by: https://github.com/moein-shariatnia/OpenAI-CLIP

import os
import random
import itertools
import argparse

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

from data import CLIPDataset
from models import CLIPModel
from utils import get_lr, AvgMeter

title = 'CLIP for documents - training'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    # env
    arg('--debug',      type=bool, default=False,  help='debug')
    arg('--dataset_path', type=str, default="Flicker-tiny",  help='dataset path')
    # train
    arg('--seed', type=int, default=987, help='random seed')
    arg('--batch_size', type=int, default=1,  help='batch size')
    arg('--data_split_valid', type=float, default=0.2,  help='how much data is valid (0.2) vs train (0.8)')
    arg('--num_workers', type=int, default=0,  help='number of workers')
    arg('--epochs', type=int, default=5,  help='number of training epochs')
    arg('--cuda', dest='cuda', action='store_true', default=True, help='Use cuda to train model')
    arg('--device_num', type=str, default=0,  help='GPU number to use')
    arg('--head_lr', type=float, default=1e-3,  help='head learnign rate')
    arg('--image_encoder_lr', type=float, default=1e-4,  help='image encoder learnign rate')
    arg('--text_encoder_lr', type=float, default=1e-5,  help='text encoder learning rate')
    arg('--weight_decay', type=float, default=1e-3,  help='weight decay')
    arg('--temperature', type=float, default=1.0,  help='temperature')
    arg('--patience', type=int, default=1,  help='patience')
    arg('--factor', type=float, default=0.8,  help='factor')
    arg('--saved_model_name', type=str, default='saved_model.pth',  help='saved model name')
    # models
    arg('--image_size', type=int, default=224,  help='image size')
    arg('--image_encoder_model_name', type=str, default="resnet50",  help='image model name')
    arg('--text_encoder_model_name', type=str, default="distilbert-base-uncased",  help='text encoder model name')
    arg('--text_tokenizer', type=str, default="distilbert-base-uncased",  help='text tokenizer')
    arg('--image_embedding_size', type=int, default=2048,  help='image embedding')
    arg('--text_embedding_size', type=int, default=768,  help='text_embedding')
    arg('--max_text_length', type=int, default=100,  help='max text length')
    arg('--pretrained', type=bool, default=True,  help='for both image encoder and text encoder')
    arg('--trainable', type=bool, default=True,  help='for both image encoder and text encoder')
    # projection head; used for both image and text encoders
    arg('--num_projection_layers', type=int, default=1,  help='number of projections layers')
    arg('--projection_dim', type=int, default=256,  help='projections dimensiones')
    arg('--dropout', type=float, default=0.1,  help='dropout')
    # testing
    arg('--query', type=str, default="white dog",  help='test: search image query')

    args = parser.parse_args()
    return args

args = get_args() # Holds all the input arguments

# setup
if torch.cuda.is_available() and args.cuda:
    device = torch.device("cuda:"+str(args.device_num))
    torch.cuda.set_device(device)
    print('Using CUDA!')
else:
    device = torch.device("cpu")

# random seeds and reproducible results:
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

np.set_printoptions(precision=2)
torch.set_printoptions(profile="full", precision=2)


def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{args.dataset_path}/captions.csv")
    max_id = dataframe["id"].max() + 1 if not args.debug else 100
    image_ids = np.arange(0, max_id)
    valid_ids = np.random.choice(
        image_ids, size=int(args.data_split_valid * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, image_size, mode):
    dataset = CLIPDataset(
        args.max_text_length,
        args.dataset_path,
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        image_size = image_size,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    # train
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(args.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, args.image_size, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, args.image_size, mode="train")

    model = CLIPModel(
        temperature=args.temperature,
        image_embedding_size=args.image_embedding_size,
        text_embedding_size=args.text_embedding_size,
        projection_dim=args.projection_dim,
        dropout = args.dropout,
        pretrained = args.pretrained,
        trainable = args.trainable,
        image_encoder_model_name=args.image_encoder_model_name,
        text_encoder_model_name=args.text_encoder_model_name
    ).to(device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": args.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": args.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": args.head_lr, "weight_decay": args.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.patience, factor=args.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), args.saved_model_name)
            print("Saved model!")
        
        lr_scheduler.step(valid_loss.avg)

if __name__ == "__main__":
    main()