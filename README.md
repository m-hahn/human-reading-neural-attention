# Modeling Human Reading with Neural Attention

Code for Modeling Human Reading with Neural Attention (EMNLP 2016).


## Preparing data

Data is expected in `data/` in the following structure: Texts in numerical form in `data/texts/`, a vocabulary in `data/dictionary.txt`, and a list of texts in `data/filenames.txt`. Samples of the appropriate format are given in the directories.


Models are saved to and loaded from `models/`.

## Creating an autoencoding model:

`th main-attention.lua 1 false false 64 50 1000 50000 5.0 false 0.1 100 0.0001 20 false none autoencoder-1 combined true 11 true 5.0 full true fixed`

or in general:

`th main-attention.lua 1 false false BATCH_SIZE SEQUENCE_LENGTH LSTM_DIMENSION VOCABULARY 5.0 false LEARNING_RATE EMBEDDINGS_DIMENSION 0.0001 20 false none NAME_OF_YOUR_MODEL combined true 11 true 5.0 full true fixed`

To control the learning rate during training, edit the file `lr-1`, whose content is the learning rate.

To control the attention rate during training, edit `attention-1` in the same directory, whose content is the attention rate (a number between 0 and 1). In the original experiments, it was initialized at 1 and annealed to 0.6.



## Creating an attention network:

`th main-attention.lua 1 false true 64 10 1000 50000 5.0 false 0.7 100 0.0001 20 false autoencoder-1 attention-1 combined true 1 true 5.0 full true fixed`

or in general:

`th main-attention.lua 1 false true BATCH_SIZE SEQUENCE_LENGTH LSTM_DIMENSION VOCABULARY TOTAL_ATTENTION_WEIGHT false LEARNING_RATE EMBEDDINGS_DIMENSION 0.1 20 false NAME_OF_AUTOENCODING_MODEL NAME_OF_ATTENTION_MODEL combined true 1 true ENTROPY_WEIGHT full true fixed`

where `TOTAL_ATTENTION_WEIGHT` is alpha, `ENTROPY_WEIGHT` is gamma from Section 4.1 of the paper.

To control the learning rate of REINFORCE during training, modify the file named `lr-att-1`, whose content is this rate (0.01 in the original experiments).

## Running an attention network to create predictions:

`th main-attention.lua 1 true true 64 10 1000 50000 5.0 false 0.7 100 0.0001 20 false attention-1 attention-1 combined false 1 true 5.0 full true fixed`

or in general:

`th main-attention.lua 1 true true BATCH_SIZE SEQUENCE_LENGTH LSTM_DIMENSION VOCABULARY TOTAL_ATTENTION_WEIGHT false LEARNING_RATE EMBEDDINGS_DIMENSION 0.1 20 false NAME_OF_ATTENTION_MODEL NAME_OF_ATTENTION_MODEL combined false 1 true ENTROPY_WEIGHT full true fixed`

This will create files with attention output in `data/annotation/`.



