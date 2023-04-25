# E. Culurciello
# April 2023
# ABRAIN web version neural network
# huggingface version


import argparse
from difflib import SequenceMatcher
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from abrain_dataset import AbrainDataset
from transformers import VisionEncoderDecoderModel


class bcolors:
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'

title = "demo ABRAIN - huggingface version"

def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    arg('--seed', type=int, default=987, help='random seed')
    arg('--batch_size', type=int, default=1,  help='batch size')
    arg('--num_workers', type=int, default=0,  help='number of workers')
    # arg('--epochs', type=int, default=200,  help='number of training epochs')
    arg('--max_target_length', default=128, help='max length of text sequence')
    arg('--vit_eye_size', type=int,default=224, help='ViT input eye size')
    arg('--image_resize', type=int,default=224, help='resize webpage image to this size')
    arg('--image_encoder_model', type=str, default='google/vit-base-patch16-224-in21k')
    arg('--text_decode_model', type=str, default='bert-base-uncased')
    arg('--trained_model_path', type=str, default="./trained_model")
    args = parser.parse_args(raw_args)
    return args


if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # model:
    model = VisionEncoderDecoderModel.from_pretrained(args.trained_model_path)
    model.eval()
    
    # dataset:
    dataset = load_dataset('../data/web')
    valid_data = AbrainDataset(dataset, args, split='validation')
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, 
                                  num_workers=args.num_workers)

    for batch in valid_dataloader:
        with torch.no_grad():
            output_ids = model.generate(batch['pixel_values'], max_length=args.max_target_length)

        preds = valid_data.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        labels = valid_data.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        preds = [p.strip() for p in preds]
        labels = [l.strip() for l in labels]
        # print(preds[0], labels[0])
        score = SequenceMatcher(None, preds[0], labels[0]).ratio()
        print(bcolors.OKBLUE + '\ntest out:\n' + bcolors.ENDC + preds[0])
        print(bcolors.OKCYAN + 'Ground truth similarity score (0-1):' + bcolors.ENDC, score)