# E. Culurciello, October 2022
# CLIP DOC pytorch lightning

# inspired by: https://github.com/moein-shariatnia/OpenAI-CLIP

import argparse

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

from data import CLIPDataset
from models import CLIPModel

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
    arg('--num_workers', type=int, default=8,  help='number of workers')
    arg('--epochs', type=int, default=3,  help='number of training epochs')
    arg('--accelerator', type=str, default='cpu', help='Accelerator to train model: cpu, gpu')
    arg('--devices', type=int, default=1,  help='number of training devices')
    # arg('--cuda', dest='cuda', action='store_true', default=True, help='Use cuda to train model')
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
pl.utilities.seed.seed_everything(seed=args.seed, workers=False)


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


def main():
    # train
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(args.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, args.image_size, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, args.image_size, mode="val")

    model = CLIPModel(
        temperature=args.temperature,
        image_embedding_size=args.image_embedding_size,
        text_embedding_size=args.text_embedding_size,
        projection_dim=args.projection_dim,
        dropout = args.dropout,
        pretrained = args.pretrained,
        trainable = args.trainable,
        image_encoder_model_name=args.image_encoder_model_name,
        text_encoder_model_name=args.text_encoder_model_name,
        image_encoder_lr=args.image_encoder_lr,
        text_encoder_lr=args.text_encoder_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        factor=args.factor,
    )

    # checkpoint_callback = ModelCheckpoint(
    #     save_top_k=1,
    #     monitor="val_loss",
    #     mode="min",
    #     dirpath="./",
    #     filename="best-model-{epoch:02d}-{val_loss:.2f}",
    #     save_weights_only=True,
    # )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        # callbacks=[checkpoint_callback],
        # enable_checkpointing=False,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    print('saving model!')
    torch.save(model.state_dict(), args.saved_model_name)

if __name__ == "__main__":
    main()