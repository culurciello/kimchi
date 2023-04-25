# E. Culurciello, October 2022
# CLIP DOC pytorch lightning


import os
from PIL import Image
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

from train import get_args, make_train_valid_dfs, build_loaders
from models import CLIPModel

title = 'CLIP for documents - testing'

args = get_args() # Holds all the input arguments

device = torch.device("cpu")

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
)#.to(device)
model.load_state_dict(torch.load(args.saved_model_name, map_location='cpu'))
model.eval()

# NOTE: I was not able to  load a checkpoint from pytorch lightning


def get_image_embeddings(valid_df):
    tokenizer = DistilBertTokenizer.from_pretrained(args.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, args.image_size, mode="valid")
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"])#.to(device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)


def find_matches(image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(args.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values)#.to(device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = torch.mm(text_embeddings_n, image_embeddings_n.T)

    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5) # pick 45 and select every 5th
    matches = [image_filenames[idx] for idx in indices[::5]] # select every 5th element
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = Image.open(f"{args.dataset_path}/images/{match}").convert('RGB')
        ax.imshow(image)
        ax.axis("off")
    
    plt.savefig("out_"+query+".png")
    plt.show()


# inference / test:
print('You are searching the image for query:', args.query)

train_df, valid_df = make_train_valid_dfs()
# note: have to use train here in small dataset, but use valid for larger!!!
# valid_df = train_df

# embed once all validation set, so do not have to do it for ecery sample!
valid_image_embeddings_file = "valid_image_embeddings.pth"
if os.path.exists(valid_image_embeddings_file):
    image_embeddings = torch.load(valid_image_embeddings_file)
else:
    image_embeddings = get_image_embeddings(valid_df)
    torch.save(image_embeddings, valid_image_embeddings_file)

find_matches(
    image_embeddings,
    query=args.query,
    image_filenames=valid_df['image'].values,
    n=9,
)