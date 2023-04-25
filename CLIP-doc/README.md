# CLIP DOC

train a contrastive model on a dataset of images and captions


### Run:

For example:

Train with "resnet50" image embedding model:
```python train.py --dataset_path ../../Datasets/Flicker-8k --batch_size 64 --num_workers 16```

Train with vision transformer large patch image embedding model:
```python train.py --dataset_path ../../Datasets/Flicker-8k --batch_size 64 --num_workers 16 --image_encoder_model_name "vit_base_patch32_224" --image_embedding_size 768```


### Results:

Train with "resnet50" image embedding model:
```Epoch: 3, train_loss=0.565, valid_loss=1.47```

Train with vision transformer large patch image embedding model:
```Epoch: 3, train_loss=0.565, valid_loss=1.06```

### Test:

Test with "resnet50" image embedding model:
``` python test.py --query "girl" --saved_model_name "saved_model_epoch3.pth"```

Test with vision transformer large patch image embedding model:
```python test.py --query "girl on beach" --saved_model_name "saved_model.pth" --dataset_path ~/Desktop/Flicker-8k --image_encoder_model_name "vit_base_patch32_224" --image_embedding_size 768```


### References:

[Ref 1](https://colab.research.google.com/drive/1hYHb0FTdKQCXZs3qCwVZnSuVGrZU2Z1w?usp=sharing)

[Ref 2](https://github.com/moein-shariatnia/OpenAI-CLIP)