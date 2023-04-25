# E. Culurciello
# April 2023
# ABRAIN web version neural network
# huggingface version

import torch
from transformers import AutoTokenizer, AutoFeatureExtractor
# import matplotlib.pyplot as plt
# import numpy as np

class AbrainDataset(torch.utils.data.Dataset):
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
        self.max_target_length = self.config.max_target_length

        self.split = split
        self.dataset = dataset[self.split]
        self.dataset_length = len(self.dataset)
        
        # image feature extractor
        self.image_encoder = AutoFeatureExtractor.from_pretrained(
                    self.config.image_encoder_model)
        # text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.text_decode_model)

    def preprocess(self, batch):
        image_in = batch['image'].convert('RGB')
        batch_image = image_in.resize((self.resize, self.resize))
        
        batch_labels = batch['label']

        model_inputs = {}

        model_inputs['labels'] = self.tokenizer(batch_labels, 
                      padding="max_length", 
                      max_length=self.max_target_length,
                      return_tensors="pt").input_ids.squeeze(0)
        
        model_inputs['pixel_values'] = self.image_encoder(
                    images=batch_image,
                    return_tensors="pt").pixel_values.squeeze(0)

        return model_inputs

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):
        """
        Load image from image_path of a webpage and its html code
        Returns:
            input_tensor : webpage image
            labels : webpage HTML code in text
        """
        sample = self.dataset[idx]
        
        return self.preprocess(sample)
