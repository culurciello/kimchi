# E. Culurciello, September 2022
# CLIP DOC 

from PIL import Image

import torch
import torchvision


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, max_text_length, dataset_path, image_filenames, captions, tokenizer, image_size):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        self.dataset_path = dataset_path
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            self.captions, padding=True, truncation=True, max_length=max_text_length
        )
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = Image.open(f"{self.dataset_path}/images/{self.image_filenames[idx]}").convert('RGB')
        item['image'] = self.transforms(image)
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)
