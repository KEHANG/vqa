import os
import numpy as np
from data import DataLoaderDisk, get_bag_of_words_embedding_matrix, get_glove_embedding_matrix
from model import vqa_model
from utils import log_to_file

############################
####### Run Training #######
############################
def train():
    # create folders needed
    if not os.path.exists('saved_models'):
      os.mkdir("saved_models")

    # create data loader
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    opt_data_train = {
        'data_root': data_path,   # MODIFY PATH ACCORDINGLY
        'fine_size': 224,
        'word_embedding_length': 1024,
        'randomize': False
        }


    data_loader = DataLoaderDisk(**opt_data_train)

    seq_length = 25
    glove_path = os.path.join(data_path, 'glove.6B', 'glove.6B.300d.txt')
    # embedding_matrix = get_bag_of_words_embedding_matrix(data_loader.tokenizer.word_index)
    embedding_matrix = get_glove_embedding_matrix(data_loader.tokenizer.word_index,
                                                  glove_path)
    model = vqa_model('vgg16', embedding_matrix, seq_length, dropout_rate=0.5, num_classes=3131)
    
    batch_size = 100
    epochs = 100
    iters = int(data_loader.train_num*epochs/batch_size)
    for iteration in range(iters):
      img_batch_train, que_batch_train, y_batch_train = data_loader.next_batch(batch_size, mode='train')
      img_batch_val, que_batch_val, y_batch_val = data_loader.next_batch(batch_size, mode='val')
      train_score = model.train_on_batch([img_batch_train, que_batch_train], y_batch_train)
      val_score = model.test_on_batch([img_batch_val, que_batch_val], y_batch_val)
      
      train_loss = float(train_score[0])
      train_acc = float(train_score[1])
      val_loss = float(val_score[0])
      val_acc = float(val_score[1])

      # log training progress
      msg = "iter = {0}, ".format(iteration)
      msg += "train loss: {0:.03f}, train acc: {1:.03f} ".format(train_loss, train_acc)
      msg += "val loss: {0:.03f}, val acc: {1:.03f}\n".format(val_loss, val_acc)
      print msg
      log_to_file(msg)

      # update lr and save model every epoch
      iter_per_epoch = int(data_loader.train_num/batch_size)
      if iteration % iter_per_epoch == 0:
        epoch = int(iteration/iter_per_epoch)
        lr = float(0.0007 * np.exp(- epoch / 30))
        model.optimizer.lr.set_value(lr)
        model.save_weights('saved_models/model_weights_epoch_{0}.h5'.format(epoch))

train()


