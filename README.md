# Image Caption Generation
Generates image captions using an encoder/decoder neural net with or without attention.
The implementation loosely follows the ideas described in the papers [1] and 
[2].

[1] Show and Tell: A Neural Image Caption Generator (Vinyals et al.) <br>
[2] Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (Xu et al.)

# Getting started

Open a terminal in the project directory and follow the steps below.

**Step 1: installation**

* install: python3, pytorch, torchvision, pytables, nltk, mosestokenizer, matplotlib, mock

**Step 2: create directory structure for output files**

```console
$ ./prepare.sh
```

**Step 3: Dowload data**

Copy the train, validation and test files to the directory 
'data/input' following the filepaths as given in 'config\_show\_tell.json'. 
We used the [Flickr30k](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/) dataset for this project.


# Runing the tests

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
$ python evaluate.py
```

The settings of a run can be configured by adapting the file 'config\_show\_tell.py' (or 'config\_show\_attend\_tell.py'). Some settings in the config file can be overwritten by passing them as command-line arguments.

**Results**
Running the 'Show\_Tell' model with the default settings on a GPU for 25 epochs took about 2 hours on the Flickr30k dataset containing 30.000 images. We achieved a BLEU-1 score of 55 on a scale of 0 to 100. Running the 'Show\_Attend\_Tell model' with the default settings on a GPU took about 5 hours for 25 epochs. We achieved a BLEU-1 score of 57.

Below we describe the subsequent steps in more detail.

# Explanation Steps

In all steps described below, use '--config config\_show\_tell.json' to use the ShowTell model [1] and use '--config config\_show\_attend\_tell.json' to use the ShowAttendTell model [2].

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

Trains a model for generating image descriptions.

optional arguments:
  --config CONFIG       Path to config file in JSON format 
  --learning_rate LEARNING_RATE
                        Sets the learning rate
  --optimizer OPTIMIZER
                        Sets the optimizer (SGD or ADAM) 
  --model MODEL         Sets the model: 'show_tell' or 'show_attend_tell'
  --alpha_c ALPHA_C     Sets the alpha regulizer in the 'show_attend_tell'
                        model. Use a value between 0 and 1.
```

```console
$ python predict.py

Generate image captions.

optional arguments:
  --config CONFIG  Path to config file in JSON format
  -i I             Path to input directory with image data
  -o O             Path to output file storing the generated captions

```

```console
$ python evaluate.py

Generates plots for train loss, validation loss and validation BLUE scores
collected during training. Calculates the BLEU score on the test data using
the trained model.

optional arguments:
  --config CONFIG  Path to config file in JSON format 
```

# Summary Project Proposal

Automatically describing the content of an image is a fundamental problem in artificial intelligence that connects computer vision and natural language processing. In [1] Vinyals et al. propose an encoder-decoder architecture for image caption generation. First a convolutional neural net (CNN) is used to encode an image into a vector, then this vector is fed into a recurrent neural network (RNN) which decodes the vector into an output sequence of words. This architecture is extended with an attention mechanism by Xu et al. in [2] to provide the decoder with more direct access to relevant parts of the image.

The attention mechanism implements the idea that words in the generated output sequence can be aligned with specific parts of the image. This idea intuitively makes sense for words that have a clear visual association such as nouns (girl, shovel, sand), numerals (three), certain adjactives (yellow) and some verbs (swimming). However, for other type of words, most notable function words (a, with, are), the alignment with image parts is less obvious or missing. Instead of being grounded by the image these words merely seem to follow from the language model itself to make the generated sentence smooth and grammatically correct.

In this project we aim to make the distinction between different reasons to generate a word explicit in the model. That is, we will implement a binary switch mechanism that allows the decoder to decide whether or not to use the image when generating the next word. We will then extend the baseline models described in [1] and [2] with this decision mechanism. We will evaluate the resulting models by comparing them to their baseline version using standard metrics used in image description generation and machine translation (BLEU, ROUGE). In addition, we will perform some experiments with the aim to gain insight into the behavior of the decision mechanism, e.g. how often and for what type of words does the model use the image?
  
[1] Show and Tell: A Neural Image Caption Generator (Vinyals et al.)
[2] Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (Xu et al.)

