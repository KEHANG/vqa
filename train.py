import os

from data import DataLoaderDisk, get_bag_of_words_embedding_matrix
from model import vqa_model

############################
####### Run Training #######
############################
def train():
    # create folders needed
    if not os.path.exists('saved_models'):
      os.mkdir("saved_models")
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
    
    batch_size = 100
    epochs = 100
    iters = int(data_loader.train_num*epochs/batch_size)
    for iteration in range(iters):
      img_batch, que_batch, y_batch = loader_train.next_batch(batch_size)
      hist = model.fit([img_batch, que_batch], y_batch, epochs=1, batch_size=batch_size)

      # save model every epoch
      iter_per_epoch = int(data_loader.train_num/batch_size)
      if iteration % iter_per_epoch == 0:
        epoch = int(iteration/iter_per_epoch)
        model.save_weights('saved_models/model_weights_epoch_{0}.h5'.format(epoch))

train()


