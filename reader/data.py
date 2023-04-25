# E. Culurciello, February 2023

import os
from PIL import Image
import pytesseract

import torch
from torchvision import transforms

# train and test directories:
train_dir = "train/"
test_dir = "test/"
dataset_filename = "dataset.pth"

# image size of text crops:
im_size = (512,512)

def get_dataset(in_dir):
    dataset = []
    for filename in os.listdir(in_dir):
        f = os.path.join(in_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if f.endswith('.png'):
                print(f'processing file: {f}')
                page = Image.open(f)
                page = page.resize(im_size)
                # display(page1)
                page_tensor = transforms.ToTensor()(page)[:3]
                transcript = pytesseract.image_to_string(page)
                # print(transcript[:500], "...")
                ascii_values = []
                for char in transcript:
                    v = ord(char) if ord(char) < 144 else 63 # limit character to 144, set to '?' others
                    ascii_values.append(v)

                transcript_tensor = torch.tensor(ascii_values)
                dataset.append( (page_tensor, transcript_tensor.unsqueeze(0)) )

    return dataset

train_data = get_dataset(train_dir)
test_data = get_dataset(test_dir)

print(f'saving dataset file: {dataset_filename}')
torch.save( (train_data, test_data), dataset_filename)
