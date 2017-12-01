import os

from data import DataLoaderDisk, get_bag_of_words_embedding_matrix
from model import vqa_model

############################
####### Run Training #######
############################
def train():
    ### read training data and validation data
    data_path = os.path.join('data')
    opt_data_train = {
        'data_root': data_path,   # MODIFY PATH ACCORDINGLY
        'fine_size': 224,
        'word_embedding_length': 1024,
        'randomize': False
        }


    loader_train = DataLoaderDisk(**opt_data_train)

    seq_length = 25
    embedding_matrix = get_bag_of_words_embedding_matrix(loader_train.tokenizer.word_index)
    model = vqa_model(embedding_matrix, seq_length, dropout_rate=0.5, num_classes=3131)
    
    img_batch, que_batch, y_batch = loader_train.next_batch(100)
    hist = model.fit([img_batch, que_batch], y_batch, epochs=1, batch_size=1)


train()


