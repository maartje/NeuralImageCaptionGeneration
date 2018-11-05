# Image Caption Generation
Generates image captions using an encoder/decoder neural net with or without attention.


# Getting started

Open a terminal in the directory 'DL_NLP' and follow the steps below.

**Step 1: installation**

* install: python3, pytorch, torchvision, pytables, nltk, mosestokenizer, matplotlib, mock

**Step 2: create directory structure for output files**

```console
$ ./prepare.sh
```

**Step 3: Dowload data**

Copy the train and test files to the directory 'data/input' following the filepaths as
given in 'config\_show\_tell.json'. We used the flickr30k dataset for this project.


# Run tests

```console
$ cd NeuralImageCaptionGeneration
$ python -m unittest discover -v
```

# Running the project

Run the following commands:

```console
$ python preprocess.py 
$ python train.py
$ python predict.py
$ python tf_idf_baseline.py
$ python evaluate.py
```

The settings of a run can be configured by adapting the file 'config\_show\_tell.py' (or 'config\_show\_attend\_tell.py'). In addition, the most important settings can be passed as command-line arguments.

Below we describe the subsequent steps in more detail.

# Explanation Steps

```console
$ python preprocess.py 

Generates vocabulary and indices vectors for captions

optional arguments:
  --config CONFIG       Path to config file in JSON format
  --min_occurences MIN_OCCURENCES
                        Sets the minimum occurrence of a word to be included
                        in the vocabulary

```

```console
$ python train.py
```
```console
$ python predict.py
```

```console
$ python evaluate.py
```

# Summary Project Proposal

Automatically describing the content of an image is a fundamental problem in artificial intelligence that connects computer vision and natural language processing. In [1] Vinyals et al. propose an encoder-decoder architecture for image caption generation. First a convolutional neural net (CNN) is used to encode an image into a vector, then this vector is fed into a recurrent neural network (RNN) which decodes the vector into an output sequence of words. This architecture is extended with an attention mechanism by Xu et al. in [2] to provide the decoder with more direct access to relevant parts of the image.

The attention mechanism implements the idea that words in the generated output sequence can be aligned with specific parts of the image. This idea intuitively makes sense for words that have a clear visual association such as nouns (girl, shovel, sand), numerals (three), certain adjactives (yellow) and some verbs (swimming). However, for other type of words, most notable function words (a, with, are), the alignment with image parts is less obvious or missing. Instead of being grounded by the image these words merely seem to follow from the language model itself to make the generated sentence smooth and grammatically correct.

In this project we aim to make the distinction between different reasons to generate a word explicit in the model. That is, we will implement a binary switch mechanism that allows the decoder to decide whether or not to use the image when generating the next word. We will then extend the baseline models described in [1] and [2] with this decision mechanism. We will evaluate the resulting models by comparing them to their baseline version using standard metrics used in image description generation and machine translation (BLEU, ROUGE). In addition, we will perform some experiments with the aim to gain insight into the behavior of the decision mechanism, e.g. how often and for what type of words does the model use the image?
  
[1] Show and Tell: A Neural Image Caption Generator (Vinyals et al.)
[2] Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (Xu et al.)

