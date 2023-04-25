# E. Culurciello
# April 2023
# ABRAIN web version neural network
# huggingface version

import random
import argparse
import numpy as np
import evaluate

import torch
from datasets import load_dataset
from abrain_dataset import AbrainDataset

from transformers import VisionEncoderDecoderModel
from transformers import TrainingArguments, Trainer


title = "train ABRAIN web - huggingface version"

def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    arg('--seed', type=int, default=987, help='random seed')
    arg('--batch_size', type=int, default=4,  help='batch size')
    arg('--num_workers', type=int, default=0,  help='number of workers')
    arg('--epochs', type=int, default=200,  help='number of training epochs')
    arg('--max_target_length', default=128, help='max length of text sequence')
    arg('--vit_eye_size', type=int,default=224, help='ViT input eye size')
    arg('--image_resize', type=int,default=224, help='resize webpage image to this size')
    arg('--image_encoder_model', type=str, default='google/vit-base-patch16-224-in21k')
    arg('--text_decode_model', type=str, default='bert-base-uncased')
    arg('--trained_model_path', type=str, default="./trained_model")
    args = parser.parse_args(raw_args)
    return args

def compute_metrics(eval_pred):
    # logits, labels = eval_pred  
    # predictions = np.argmax(logits[0], axis=-1)
    # return metric.compute(predictions=predictions, references=labels)
    return {'precision': 0.0}

if __name__ == "__main__":
    args = get_args() # Holds all the input arguments

    # setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # random seeds and reproducible results:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.set_printoptions(precision=2)
    torch.set_printoptions(profile="full", precision=2)


    # train metric:
    metric = evaluate.load("accuracy")

    # dataset:
    dataset = load_dataset('../data/web')
    train_data = AbrainDataset(dataset, args, split='train')
    valid_data = AbrainDataset(dataset, args, split='validation')

    # model:
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        args.image_encoder_model, args.text_decode_model)
    
    model.config.decoder_start_token_id = train_data.tokenizer.cls_token_id
    model.config.pad_token_id = train_data.tokenizer.pad_token_id

    # test model:
    # batch = train_data[0]
    # print(len(train_data))
    # print("TEST model:")
    # print(batch['pixel_values'].shape)
    # print(batch['labels'])
    # out = model(pixel_values=batch['pixel_values'].unsqueeze(0), labels=batch['labels'].unsqueeze(0))
    # print("TEST loss:", out.loss)
    # print(out.logits.shape)
    
    # training:
    training_args = TrainingArguments(
            seed=args.seed,
            output_dir="test_trainer", 
            evaluation_strategy="epoch",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            # report_to="tensorboard",
        )
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=valid_data,
            compute_metrics=compute_metrics,
        )

    trainer.train()
    trainer.save_model(args.trained_model_path)
