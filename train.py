import os

from data import DataLoaderDisk

data_path = os.path.join('..', 'data')
opt_data_train = {
    'data_root': data_path,   # MODIFY PATH ACCORDINGLY
    'fine_size': 224,
    'word_embedding_length': 1024,
    'randomize': False
    }


loader_train = DataLoaderDisk(**opt_data_train)

img_batch, que_batch, y_batch = loader_train.next_batch(100)