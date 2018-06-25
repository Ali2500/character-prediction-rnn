# character-prediction-rnn
A vanilla RNN implementation, along with accompanying code to generate training/validation data from the Reuters21578 dataset.

The Reuters21578 dataset can be downloaded from here:
http://www.daviddlewis.com/resources/testcollections/reuters21578/

The RNN model is composed of standard LSTM cells and is trained with dropout. The input text strings are encoded with one-hot encoding; the current encoding scheme allows for a total of 96 distinct characters (all printable ASCII characters in the range 32 - 127, and the newline character).

## Dependencies

The model is implemented using TensorFlow. In addition, the Beautiful Soup Python library is used for parsing the .sgm files provided by the Reuters dataset. This library can be installed simply by running:

`sudo apt-get install python-bs4`

## Initial Setup

Before running any of the scripts, it is required to correctly initialize two variables in definitions.py according to your setup:

1. MODEL_SAVE_DIR: This is where the model will be saved to when training, and loaded from during inference.
2. DATASET_DIR: This should point to the path where you have placed the reuters21578 dataset files downloaded from the link above.
3. LOG_DIR: For each training session, a time-stamped sub-directory will be created in side this directory and all the log files generated during training will be written to it.

Once definitions.py has been correctly configured, run the data_parser.py script to parse the dataset. This script takes two optional arguments: (1) the ratio of the number of articles to reserve for validation to the total number of articles, and the minimum acceptable article length (i.e. articles that have fewer characters are not considered). Running this script will create a subdirectory called 'processing' in your dataset directory containing two pickled files containing the parsed articles.

## Training

With the data correctly parsed, you can run the 'train_rnn.py' script to start the training session. This script takes the following optional arguments:

1. epochs: The number of epochs to train to.
2. batch-size: The size of each mini-batch used for training
3. seq-length: The size of each character string used for training.
4. char-skip: The gap between successive character strings given to the network.

Additionally, there are 4 variables in the script that are worth elaborating:

1. TEXT_PREDICTION_LOG_INTERVAL: After every this many training mini-batches, the prediction performed by the network for the last minibatch will be written to the file in the timestamped log directory. This can be helpful to get a feel for the progress being made by the training process.
2. MODEL_SAVE_INTERVAL: After every this many training mini-batches, the model will be written to MODEL_SAVED_DIR.
3. VALIDATION_INTERVAL: After every this many training mini-batches, inference will be run on the validation set and the loss and accuracy will be recorded.

Note that if training if canceled mid-way, it will resume from the last saved model checkpoint by default (unless the '--train-from-scratch' flag is explicitly set).

## Inference

A pre-trained model (trained up to 100 epochs) can found here: https://drive.google.com/open?id=1Bh2bge762YMGUHh_XNsB10jgp-QHm2SW. 

To use it, simply extract the zip archive to where the 'MODEL_SAVE_DIR' variable in definitions.py is set to.

Inference is performed by loading strings from a plain text file. Each line in the string is assumed to be a separate sample which will be extended by the RNN. The inference.py script therefore takes two arguments:

1. text-file (shorthand: -i): The path to the file containing the string samples
2. prediction-length: The number of characters to predict after the end of each sample.
3. output (optional): Path to output file to which the predictions will be written.
