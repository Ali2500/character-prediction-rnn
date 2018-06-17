# character-prediction-rnn
A vanilla RNN implementation, along with accompanying code to generate training/validation data from the Reuters21578 dataset.

The Reuters21578 dataset can be downloaded from here:
http://www.daviddlewis.com/resources/testcollections/reuters21578/

Before running any of the scripts, make sure to set the directory paths in definitions.py. These paths specify where the dataset is located, where the trained model will be saved etc.

The RNN model is composed of standard LSTM cells and is trained with dropout. The input text strings are encoded with one-hot encoding; the current encoding scheme allows for a total of 96 distinct characters (all printable ASCII characters in the range 32 - 127, and the newline character).
