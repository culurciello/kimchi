# E. Culurciello, October 2022
# CLIP DOC pytorch lightning 

import itertools

import torch
from torch import nn
import torch.nn.functional as F

import timm
import pytorch_lightning as pl
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

# Encode images to a fixed size vector
class ImageEncoder(nn.Module):
    def __init__(
        self, model_name, pretrained, trainable):
        super().__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0, 
            # global_pool="avg", # EC removed for use with ViT
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

# Encode text to a fixed size vector
class TextEncoder(nn.Module):
    def __init__(self, model_name, pretrained, trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(pl.LightningModule):
    def __init__(
        self,
        temperature,
        image_embedding_size,
        text_embedding_size,
        projection_dim,
        dropout,
        pretrained,
        trainable,
        image_encoder_model_name,
        text_encoder_model_name,
        image_encoder_lr,
        text_encoder_lr,
        head_lr,
        weight_decay,
        patience,
        factor,
    ):
        super().__init__()
        self.temperature = temperature
        self.image_embedding_size = image_embedding_size
        self.text_embedding_size = text_embedding_size
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.pretrained = pretrained
        self.trainable = trainable
        self.image_encoder_model_name = image_encoder_model_name
        self.text_encoder_model_name = text_encoder_model_name
        self.image_encoder_lr = image_encoder_lr
        self.text_encoder_lr = text_encoder_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.factor = factor

        self.image_encoder = ImageEncoder(self.image_encoder_model_name, self.pretrained, self.trainable)
        self.text_encoder = TextEncoder(self.text_encoder_model_name, self.pretrained, self.trainable)

        self.image_projection = ProjectionHead(
            embedding_dim=self.image_embedding_size,
            projection_dim=self.projection_dim,
            dropout=self.dropout)

        self.text_projection = ProjectionHead(
            embedding_dim=self.text_embedding_size,
            projection_dim=self.projection_dim,
            dropout=self.dropout)

    def configure_optimizers(self):
        params = [
            {"params": self.image_encoder.parameters(), "lr": self.image_encoder_lr},
            {"params": self.text_encoder.parameters(), "lr": self.text_encoder_lr},
            {"params": itertools.chain(
                self.image_projection.parameters(), self.text_projection.parameters()
            ), "lr": self.head_lr, "weight_decay": self.weight_decay}
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=0.)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self.patience, factor=self.factor
        )
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = torch.mm(image_embeddings, image_embeddings.T)
        texts_similarity = torch.mm(text_embeddings, text_embeddings.T)
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        self.log("%s_loss" % mode, loss.mean()) # batch_size=batch_size)
        return loss.mean()


    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")