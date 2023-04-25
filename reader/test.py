# This is a model of a foveating neural network
# it reads text from images and produces a list of ascii characters

# E. Culurciello, February 2023

from difflib import SequenceMatcher
import torch
from model import ReaderS2S


class bcolors:
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'


def test(model, test_dataset):
    model.eval()
    for (inputs, targets) in test_dataset:
        with torch.no_grad():
            out = model(inputs, targets)
        
        out = out.squeeze(0).squeeze(0)
        
        read_string = ''
        target_string = ''
        for c in out:
            read_string += chr(c.argmax())

        for c in targets.squeeze(0):
            target_string += chr(c.item())

        score = SequenceMatcher(None, read_string, target_string).ratio()
        print(bcolors.OKBLUE + '\ntest:\n' + bcolors.ENDC + read_string)
        print(bcolors.OKCYAN + 'Ground truth similarity score (0-1):' + bcolors.ENDC, score)

# test model:
model = ReaderS2S()
model.load_state_dict(torch.load('model.pth')) # load trained model weights
# get dataset:
dataset_filename = "dataset.pth"
_, test_dataset = torch.load(dataset_filename)
test(model, test_dataset)