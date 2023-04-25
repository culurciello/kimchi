# CLIP DOC

train a contrastive model on a dataset of images and captions

[PyTorch lightining](https://pytorch-lightning.readthedocs.io/) version


### Run:

For example:

Train with "resnet50" image embedding model:
```python train.py --dataset_path ../../Datasets/Flicker-8k --batch_size 64 --num_workers 16```

Train with vision transformer large patch image embedding model:
```python train.py --dataset_path ../../Datasets/Flicker-8k --batch_size 64 --num_workers 16 --image_encoder_model_name "vit_base_patch32_224" --image_embedding_size 768```


### Results:

Train with "resnet50" image embedding model:
```Epoch: 3, train_loss=0.565, valid_loss=0.36```

Train with vision transformer large patch image embedding model:
```Epoch: 3, train_loss=0.565, valid_loss=1.06```

### Test:

Test with "resnet50" image embedding model:
``` python test.py --dataset_path ~/Desktop/Flicker-8k --query "girl"```

Test with vision transformer large patch image embedding model:
```python test.py --dataset_path ~/Desktop/Flicker-8k --image_encoder_model_name "vit_base_patch32_224" --image_embedding_size 768 --query "girl on beach"```


### References:

[Ref 1](https://colab.research.google.com/drive/1hYHb0FTdKQCXZs3qCwVZnSuVGrZU2Z1w?usp=sharing)

[Ref 2](https://github.com/moein-shariatnia/OpenAI-CLIP)