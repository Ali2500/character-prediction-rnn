import os

N_CHARS = 96

MODEL_SAVE_DIR = os.path.join(os.path.expanduser('~'), 'character-prediction-rnn/saved-models')
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

MODEL_PREFIX = 'character-prediction-rnn'

DATASET_DIR = os.path.join(os.path.expanduser('~'), 'datasets/reuters21578')
if not os.path.exists(DATASET_DIR):
    raise EnvironmentError("The 'reuters21578' dataset directory was not found at the pre-set path. Consider reviewing "
                           "the paths in 'definitions.py' for correctness")

LOG_DIR = os.path.join(os.path.expanduser('~'), 'character-prediction-rnn/logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
