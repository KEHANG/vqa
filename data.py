import os
import json
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

MAX_SEQUENCE_LENGTH = 25
MAX_NB_WORDS = 20000

def load_data(data_root):

    question_train_file = os.path.join(data_root, 'Questions_Train_abstract_v002/MultipleChoice_abstract_v002_train2015_questions.json')

    question_train_dict = json.load(open(question_train_file))

    x_train_dict = {}
    for item in question_train_dict['questions']:
        image_id = item['image_id']
        question_id =  item['question_id']
        key = "{0}_{1}".format(image_id, question_id)
        x_train_dict[key] = item['question']

    question_val_file = os.path.join(data_root, 'Questions_Val_abstract_v002/MultipleChoice_abstract_v002_val2015_questions.json')

    question_val_dict = json.load(open(question_val_file))

    x_val_dict = {}
    for item in question_val_dict['questions']:
        image_id = item['image_id']
        question_id =  item['question_id']
        key = "{0}_{1}".format(image_id, question_id)
        x_val_dict[key] = item['question']

    ans_train_file = os.path.join(data_root, 'Questions_Train_abstract_v002/abstract_v002_train2015_annotations.json')

    ans_train_dict = json.load(open(ans_train_file))

    y_train_dict = {}
    for item in ans_train_dict['annotations']:
        answers = item['answers']
        histo_ans = {}
        for ans in answers:
            if ans['answer'] not in histo_ans:
                histo_ans[ans['answer']] = 1
            else:
                histo_ans[ans['answer']] += 1

        top_ans = sorted(histo_ans.items(), key=lambda tup:tup[1], reverse=True)[0][0]
        
        image_id = item['image_id']
        question_id =  item['question_id']
        key = "{0}_{1}".format(image_id, question_id)
        y_train_dict[key] = top_ans

    ans_val_file = os.path.join(data_root, 'Questions_Val_abstract_v002/abstract_v002_val2015_annotations.json')

    ans_val_dict = json.load(open(ans_val_file))

    y_val_dict = {}
    for item in ans_val_dict['annotations']:
        answers = item['answers']
        histo_ans = {}
        for ans in answers:
            if ans['answer'] not in histo_ans:
                histo_ans[ans['answer']] = 1
            else:
                histo_ans[ans['answer']] += 1

        top_ans = sorted(histo_ans.items(), key=lambda tup:tup[1], reverse=True)[0][0]
        
        image_id = item['image_id']
        question_id =  item['question_id']
        key = "{0}_{1}".format(image_id, question_id)
        y_val_dict[key] = top_ans

    histo_all_answer = {}
    for _, ans in y_train_dict.iteritems():
        if ans not in histo_all_answer:
            histo_all_answer[ans] = 1
        else:
            histo_all_answer[ans] += 1

    for _, ans in y_val_dict.iteritems():
        if ans not in histo_all_answer:
            histo_all_answer[ans] = 1
        else:
            histo_all_answer[ans] += 1

    topK_tups = sorted(histo_all_answer.items(), key=lambda tup:tup[1], reverse=True)

    selected_ans_list = [tup[0] for tup in topK_tups]

    ximgid_train = []
    xque_train = []
    y_train = []
    for key in y_train_dict:
        y_ans = y_train_dict[key]
        y_ans = selected_ans_list.index(y_ans)
        x_que = x_train_dict[key]
        x_img_id = key.split('_')[0]
        
        ximgid_train.append(x_img_id)
        xque_train.append(x_que)
        y_train.append(y_ans)

    ximgid_val = []
    xque_val = []
    y_val = []
    for key in y_val_dict:
        y_ans = y_val_dict[key]
        y_ans = selected_ans_list.index(y_ans)
        x_que = x_val_dict[key]
        x_img_id = key.split('_')[0]
        
        ximgid_val.append(x_img_id)
        xque_val.append(x_que)
        y_val.append(y_ans)

    return (ximgid_train, xque_train, y_train), (ximgid_val, xque_val, y_val),selected_ans_list

def load_image(img_path, size=None):

    img_obj = Image.open(img_path)
    
    if size:
        img_obj = img_obj.resize(size)
    
    img_arry = np.array(img_obj)
    img_arry = img_arry[:,:,:3]
    
    return img_arry

def get_tokenizer(texts):

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)

    return tokenizer

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.data_root = os.path.join(kwargs['data_root'])
        self.fine_size = kwargs['fine_size']
        self.word_embedding_length = kwargs['word_embedding_length']
        self.randomize = kwargs['randomize']
        self.data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

        # read data info
        self.ximg_path_train = []
        (ximgid_train, xque_train, y_train), (ximgid_val, xque_val, y_val), selected_ans_list = load_data(self.data_root)
        for ximgid in ximgid_train:
            img_path = 'scene_img_abstract_v002_train2015/abstract_v002_train2015_{0:012d}.png'.format(int(ximgid))
            self.ximg_path_train.append(os.path.join(self.data_root, img_path))

        self.ximg_path_train = np.array(self.ximg_path_train, np.object)
        self.xque_train = np.array(xque_train, np.object)
        self.y_train = np.array(y_train, np.int64)

        # get tokenize for xque
        all_que = xque_train + xque_val
        self.tokenizer = get_tokenizer(all_que)

        self.selected_ans_list = selected_ans_list
        self.num = self.ximg_path_train.shape[0]
        print('# Training examples found: {0}'.format(self.num))

        # permutation
        if self.randomize:
            perm = np.random.permutation(self.num) 
            self.ximg_path_train[:, ...] = self.ximg_path_train[perm, ...]
            self.xque_train[:] = self.xque_train[perm, ...]
            self.y_train[:] = self.y_train[perm, ...]

        self._idx = 0
        
    def next_batch(self, batch_size):

        img_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3))
        que_batch = np.array(['']*batch_size, np.object)
        y_batch = np.zeros(batch_size)

        desired_img_shape = (self.fine_size, self.fine_size)
        for i in range(batch_size):
            image = load_image(self.ximg_path_train[self._idx], desired_img_shape)
            image = image.astype(np.float32)/255.
            image = image - self.data_mean
            
            img_batch[i, ...] =  image
            que_batch[i, ...] = self.xque_train[self._idx]
            y_batch[i, ...] = self.y_train[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
        
        que_batch = self.tokenizer.texts_to_sequences(que_batch)
        que_batch = pad_sequences(que_batch, maxlen=MAX_SEQUENCE_LENGTH)
        y_batch = keras.utils.to_categorical(y_batch, len(self.selected_ans_list))
        
        return img_batch, que_batch, y_batch


