# Reader - an OCR model

frustrated by the lack of working OCR models, we are implementing one out of first principles with all transformers:

- visual transformer for input images
- input images are turned into a sequence of visual embeddings (like sequential reading foveation)
- Transformer encoder-decoder to generate text
- implements a sequence-to-sequence model

## generate dataset

Uses a directory train/ and test/ filled with png files of text, and produces pytorch image, text pairs for tranining

`python data.py`

Note: ground truth targets are generated with OCR (PyTesseract)


## train

`python data.py`

## test

`python test.py`


###

Note: If pytesseract does not work for you, you can get the dataset from data.py [here](https://drive.google.com/file/d/1ZWi4n5Q_4FoDKQsWgaOtIRrSakT7Im7D/view?usp=sharing).
