# E. Culurciello, A. Chang
# March 2023
# ABRAIN version neural network model for forms

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict 
from transformers import AutoProcessor
# from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import evaluate
import math
import numpy as np
import re
from pathlib import Path
from typing import Any, List, Tuple
import random
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk import edit_distance
from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers import VisionEncoderDecoderConfig
from demo import get_bboxes
from PIL import Image, ImageDraw, ImageFont

MAX_PATCHES = 1024
MAX_LEN = 200
image_size = [640, 480]

# update image_size of the encoder
# during pre-training, a larger image size was used
model_config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
model_config.encoder.image_size = image_size  # (height, width)
# update max_length of the decoder (for generation)
model_config.decoder.max_length = MAX_LEN

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
DonutModel = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=model_config)

processor.feature_extractor.size = image_size[::-1]
processor.feature_extractor.do_align_long_axis = False

def gen_num_tokens(max_num = 2800):
    return ['<' + str(x) + '>' for x in range(max_num)]

class DonutDataset_bbox(Dataset):
    def __init__(
        self,
        dataset: str,
        max_length: int = MAX_LEN,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s_docvqa>",
        prompt_end_token: str = "<s_answer>",
 
    ):
        super().__init__()

        self.added_tokens = []
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
    

        self.dataset = dataset
        self.dataset_length = len(self.dataset)

        additional_tokens = gen_num_tokens() + ['<x>', '<y>', '<w>', '<h>', '<s>', '</s>']

        self.add_tokens(additional_tokens)
        self.add_tokens([self.task_start_token, self.prompt_end_token])

        self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)


    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            DonutModel.decoder.resize_token_embeddings(len(processor.tokenizer))
            self.added_tokens.extend(list_of_tokens)

    def __len__(self) -> int:
        return self.dataset_length - 1

    def __getitem__(self, idx: int):
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]

        # input_tensor
        pixel_values = processor(sample["image"].convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        input_tensor = pixel_values.squeeze()

        # input_ids can be more than one, e.g., DocVQA Task 1
        processed_parse = sample["label"]
        input_ids = processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)


        if self.split == "train":
            labels = input_ids.clone()
            labels[labels == processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
            labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(input_ids == self.prompt_end_token_id).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse, idx
                    

class DonutModel_Trainer(pl.LightningModule):
    def __init__(self, config, max_length=128):
        super().__init__()

        self.max_length = max_length
        self.config = config
        self.processor = processor
        self.model = DonutModel

    def training_step(self, batch, batch_idx):
        pixel_values, decoder_input_ids, labels = batch
        outputs = self.model(pixel_values,
                             decoder_input_ids=decoder_input_ids[:, :-1],
                             labels=labels[:, 1:])
        loss = outputs.loss
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, decoder_input_ids, prompt_end_idxs, answers, idx = batch
        decoder_prompts = pad_sequence(
            [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
            batch_first=True,)
        outputs = self.model.generate(pixel_values,
                                      decoder_input_ids=decoder_prompts,
                                      max_length=self.max_length,
                                      early_stopping=True,
                                      pad_token_id=self.processor.tokenizer.pad_token_id,
                                      eos_token_id=self.processor.tokenizer.eos_token_id,
                                      use_cache=True,
                                      num_beams=1,
                                      bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                      return_dict_in_generate=True,)

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            # remove first task start token
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()
            predictions.append(seq)

        scores = list()
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) /
                          max(len(pred), len(answer)))

            if (self.config.get("verbose", False) and len(scores) == 1) or (batch_idx == 1):
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        return scores

    def validation_epoch_end(self, validation_step_outputs):
        # I set this to 1 manually
        # (previously set to len(self.config.dataset_name_or_paths))
        num_of_loaders = 1
        if num_of_loaders == 1:
            validation_step_outputs = [validation_step_outputs]
        assert len(validation_step_outputs) == num_of_loaders
        cnt = [0] * num_of_loaders
        total_metric = [0] * num_of_loaders
        val_metric = [0] * num_of_loaders
        for i, results in enumerate(validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
        self.log_dict({"val_metric": np.sum(total_metric) /
                      np.sum(cnt)}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.get("lr"))
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                     "monitor": "val_metric"}
        return [optimizer], [scheduler]


def main():

    rdataset = load_dataset('achang/form_bbox')
    # rdataset = load_dataset('./data/')
    dataset = DatasetDict({
        "train": rdataset['train'],
        "validation": rdataset['train']#.select(range(80)),
    })

    train_dataset = DonutDataset_bbox(dataset['train'], split="train")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)

    val_dataset = DonutDataset_bbox(dataset['validation'], split="validation")
    valid_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1)

    config = {
        "max_epochs": 100,
        "gradient_clip_val": 1.0,
        "lr": 3e-5,
        "train_batch_sizes": [4],
        "verbose": False,
    }
    tb_logger = TensorBoardLogger("tb_logs", name="my_donut_model")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        filename="artifacts",
        save_top_k=2,
        save_last=False,
        mode="min",
    )

    model_module = DonutModel_Trainer(config)

        
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.get("max_epochs"),
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model_module,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader)

    print('finished saving model')

def demo():
    config = {
        "max_epochs": 100,
        "gradient_clip_val": 1.0,
        "lr": 1e-5,
        "train_batch_sizes": [4],
        "verbose": False,
    }
    
    dataset = load_dataset('achang/form_bbox')
    train_dataset = DonutDataset_bbox(dataset['train'], split="validation")
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1)

    model_name = "./donutform_model_0/checkpoints/artifacts.ckpt"
    dmodel = DonutModel_Trainer(config)
    checkpoint = torch.load(model_name)
    dmodel.load_state_dict(checkpoint["state_dict"])
    dmodel = dmodel.eval()

    cnt = 0
    for batch in train_dataloader:
        pixel_values, decoder_input_ids, prompt_end_idxs, answers, idx = batch
        decoder_prompts = pad_sequence(
            [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
            batch_first=True,)
        outputs = dmodel.model.generate(pixel_values,
                                      decoder_input_ids=decoder_prompts,
                                      max_length=dmodel.max_length,
                                      early_stopping=True,
                                      pad_token_id=dmodel.processor.tokenizer.pad_token_id,
                                      eos_token_id=dmodel.processor.tokenizer.eos_token_id,
                                      use_cache=True,
                                      num_beams=1,
                                      bad_words_ids=[[dmodel.processor.tokenizer.unk_token_id]],
                                      return_dict_in_generate=True,)

        predictions = []
        for seq in dmodel.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(dmodel.processor.tokenizer.eos_token, "").replace(dmodel.processor.tokenizer.pad_token, "")
            # remove first task start token
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()
            predictions.append(seq)

        scores = list()
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(dmodel.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) /
                          max(len(pred), len(answer)))

            print(f"Prediction: {pred}")
            print(f"    Answer: {answer}")
            print(f" Normed ED: {scores[0]}")
            boxes = get_bboxes(pred)

            image = train_dataset.dataset[idx.item()]['image'].convert('RGB')
            draw = ImageDraw.Draw(image)
            for box in boxes:    
                actual_box = [box['x'], box['y'], box['x']+box['w'], box['y']+box['h']]
                draw.rectangle(actual_box, outline='blue')
            image.show()


        cnt += 1
        if cnt == 3:
            break

if __name__ == "__main__":
    # main()
    demo()
