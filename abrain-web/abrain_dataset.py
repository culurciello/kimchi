# E. Culurciello
# March 2023
# ABRAIN web version neural network

import torch
from torch.utils.data import Dataset
import torchvision
import torchtext
from transformers import BertTokenizer
import timm

# import matplotlib.pyplot as plt
# import numpy as np

class AbrainDataset(Dataset):
    """
    Abrain Dataset: 
    ec/webpages/
    dict {webpage image seq foveation, html text code}
    """

    def __init__(
        self,
        dataset,
        config,
        split: str = "train",
    ):
        super().__init__()
        self.config = config
        self.resize = self.config.image_resize
        self.eye_size = self.config.vit_eye_size
        self.max_length = self.config.max_target_length
        self.ignore_id = -100

        self.split = split
        self.dataset = dataset[self.split]
        self.dataset_length = len(self.dataset)

        # ViT page encoder:
        self.image_encoder = timm.create_model('vit_base_patch16_224', pretrained=False) # load ViT-B/16
        self.image_encoder.head = torch.nn.Identity() # remove the classification head

        self.image_transformation = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.resize, self.resize)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # convert to sequence of foveation and ViT encoding
    def img2seq_crop(self, image_in):
        # convert PIl to tensor and transform
        image_tensor = self.image_transformation(image_in).float().unsqueeze(0)
        # prepare image as sequence of foveations (crops)
        img_seq = []
        for j in range(0, image_tensor.shape[3]-self.eye_size, self.eye_size//2):
            for i in range(0, image_tensor.shape[2]-self.eye_size, self.eye_size//2):
                image_crop = torchvision.transforms.functional.crop(image_tensor, j, i, self.eye_size, self.eye_size)
                # debug image
                # print(image_crop.shape)
                # plt.imshow(image_crop.squeeze(0).numpy().transpose(1,2,0))
                # plt.show()
                img_seq.append(self.image_encoder(image_crop))

        img_seq = torch.cat(img_seq, dim=0)
        return img_seq 

    def preprocess(self, batch):
        # process image:
        image_seq = self.img2seq_crop(batch['image'].convert('RGB'))
        # debug:
        # print(batch['label'])
        # if char then we want to use ascii and learn characters
        # else we use a full BERT vocab
        if self.config.char:
            ascii_values = []
            for char in batch['label']:
                v = ord(char) if ord(char) < 144 else 63 # limit character to 144, set to '?' others
                ascii_values.append(v)

            labels = torch.tensor(ascii_values)
        else:
            tokens = self.tokenizer(batch['label'])
            labels = torch.tensor(tokens['input_ids'])

        return image_seq, labels

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : crop / foveation image sequence
            labels : webpage HTML code in text
        """
        sample = self.dataset[idx]
        
        return self.preprocess(sample)
