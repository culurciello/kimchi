# This is a model of a foveating neural network
# it reads text from images and produces a list of ascii characters

# E. Culurciello, February 2023
import torch
import torch.nn as nn
from model import ReaderS2S


class bcolors:
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'


def train(model, criterion, optimizer, train_dataset):
    model.train()
    for epoch in range(10):
        for (inputs, targets) in train_dataset:
            optimizer.zero_grad()
            out = model(inputs, targets)
            loss = criterion(out.squeeze(0), targets.squeeze(0))
            loss.backward()
            optimizer.step()
        
        print(f'epoch {epoch}, loss {loss.item()}')

    # save model
    model_file = 'model.pth'
    print(f'saving model to: {model_file}')
    torch.save(model.state_dict(), model_file)

# train model
model = ReaderS2S()
# model = torch.compile(model) # PyTorch 2.0 speed upgrades
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
# get dataset:
dataset_filename = 'dataset.pth'
train_dataset, _ = torch.load(dataset_filename)
train(model, criterion, optimizer, train_dataset)
