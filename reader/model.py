# E. Culurciello, February 2023

import torch
import torch.nn as nn
from torchvision import transforms
import timm


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class ReaderS2S(nn.Module):
    def __init__(self, im_size=224, ascii_num=144, src_mask_sz=9, tgt_mask_sz=1367):
        super(ReaderS2S, self).__init__()
        self.im_size = im_size
        self.page_encoder = timm.create_model('vit_base_patch16_224', pretrained=False) # load ViT-B/16
        self.page_encoder.head = nn.Identity() # remove the classification head
        # self.src_mask = generate_square_subsequent_mask(src_mask_sz)
        # self.tgt_mask = generate_square_subsequent_mask(tgt_mask_sz)
        self.transformer = nn.Transformer(d_model=768,
                                          nhead=4, 
                                          num_encoder_layers=1,
                                          num_decoder_layers=1,
                                          dim_feedforward=768,
                                          batch_first=True)
        self.out_embedder = nn.Embedding(144, 768) # need to embed these characters ascii code to 768 dim
        self.decode_out = nn.Linear(768, ascii_num)

    def forward(self, image_in, seq_out):
        self.tgt_mask = generate_square_subsequent_mask(seq_out.shape[1])
        # scan image by moving encoder ViT by 224/2 pixels
        encoder_page_features = []
        for j in range(0, image_in.shape[1]-self.im_size, self.im_size//2):
            for i in range(0, image_in.shape[2]-self.im_size, self.im_size//2):
                image_crop = transforms.functional.crop(image_in, i, j, self.im_size, self.im_size)
                encoder_page_features.append(self.page_encoder(image_crop.unsqueeze(0)))

        # this is the encoder page sequence:
        encoder_page_features = torch.cat(encoder_page_features, dim=0).unsqueeze(0)
        # print(encoder_page_features.shape, seq_out.shape)

        # send to sequence transformer encoder
        seq_out_emb = self.out_embedder(seq_out)
        encoded_seq = self.transformer(encoder_page_features, seq_out_emb, 
                                    # src_mask=self.src_mask, # this not needed: want to attend to all input image
                                    tgt_mask=self.tgt_mask)
        output_seq = self.decode_out(encoded_seq)
        
        return output_seq