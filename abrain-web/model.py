# E. Culurciello, A. Chang
# March 2023
# ABRAIN web version neural network

# ABRAIN model v1.0

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random

# import pytorch_lightning as pl

def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# class ABrainV1(pl.LightningModule):
class ABrainV1(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.dev = device
        self.config = config
        self.max_length = self.config.max_target_length

        # text embedder:
        if self.config.char:
            self.text_encoder = nn.Embedding(self.config.ascii_num, self.config.d_size)
            self.text_decoder = nn.Linear(self.config.d_size, self.config.ascii_num)
        else:
            self.text_encoder = nn.Embedding(self.config.vocab_size, self.config.d_size)
            self.text_decoder = nn.Linear(self.config.d_size, self.config.vocab_size)

        self.positional_encoding = PositionalEncoding(self.config.d_size, dropout=self.config.dropout)
        # reduce from 768 ViT to 512 for main transformer!
        self.vision_encoder_reducer = nn.Linear(self.config.vit_size, self.config.d_size)
        
        # Main Trasformer decoder:
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.config.d_size, nhead=self.config.nheads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.max_target_length)

        self.decode_out = nn.Linear(768, self.config["ascii_num"])

    def forward(self, seq_in, seq_out):
        self.tgt_mask = generate_square_subsequent_mask(seq_out.shape[1]).to(self.dev)
        out_seq_emb = self.text_encoder(seq_out)
        tgt_emb = self.positional_encoding(out_seq_emb)
        sin_reduced = self.vision_encoder_reducer(seq_in)
        encoded_seq = self.transformer_decoder(tgt_emb, sin_reduced,
                                    # src_mask=self.src_mask, # this not needed: want to attend to all input image
                                    tgt_mask=self.tgt_mask)
        output_seq = self.text_decoder(encoded_seq)
        return output_seq
    
    def generate(self, seq_in, first_token, max_length=128):
        # seq_out = torch.ones(1, 1, device=self.dev, dtype=torch.long)*random.randint(0, 144)
        seq_out = first_token
        for _ in range(max_length):
            pred = self(seq_in, seq_out)
            next_item = pred.topk(1)[1].view(-1)[-1].item() 
            next_item = torch.tensor([[next_item]], device=self.dev)
            seq_out = torch.cat((seq_out, next_item), dim=1)
        
        return seq_out

    def training_step(self, batch):
        image_in, seq_out = batch
        encoded_seq, seq_out_emb = self(image_in, seq_out)
        print(encoded_seq.shape, seq_out_emb.shape)
        loss = F.cross_entropy(encoded_seq, seq_out_emb)
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def generate(self, seq_in, first_tokens, max_length=128):
        output_seq = torch.zeros(1, max_length, device=self.dev, dtype=torch.long)
        ts = first_tokens.shape[1]
        output_seq[0,0:ts] = first_tokens
        sin_reduced = self.vision_encoder_reducer(seq_in) # this only needs to run once
        for t in range(ts, max_length-1):
            out_seq_emb = self.text_encoder(output_seq)
            tgt_emb = self.positional_encoding(out_seq_emb)
            encoded_seq = self.transformer_decoder(tgt_emb[:,0:t], sin_reduced)
            logits = self.text_decoder(encoded_seq)
            probs = F.softmax(logits, dim=-1)
            # non-greedy decoding using sampling from prob distribution:
            idx_next = torch.multinomial(probs[0,-1], num_samples=1)
            output_seq[0,t+1] = idx_next
        
        return output_seq

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                     "monitor": "val_metric"}
        return [optimizer], [scheduler]
