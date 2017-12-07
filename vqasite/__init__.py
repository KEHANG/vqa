import os
import shutil
import numpy as np
from flask import Flask, render_template, request
from keras.preprocessing.sequence import pad_sequences

from model import vqa_model
from data import DataLoaderDisk, get_embedding_matrix, load_image, MAX_SEQUENCE_LENGTH

app = Flask(__name__)

root = os.path.dirname(os.path.dirname(__file__))

# load data_loader and model

embedding_type = 'glove'
embedding_dim = 300
image_model_name = 'vgg16'
weights_file = os.path.join(root, 'evaluation', 'vqa-vgg16-glove300d', 'model_weights_epoch_11.h5')

data_path = os.path.join(root, 'data')
option = {
    'data_root': data_path,   # MODIFY PATH ACCORDINGLY
    'fine_size': 224,
    'word_embedding_length': 1024,
    'randomize': False
    }
data_loader = DataLoaderDisk(**option)

word_index = data_loader.tokenizer.word_index

embedding_path = ''
if embedding_type == 'glove':
    embedding_path = os.path.join(data_path, 'glove.6B', 
                                  'glove.6B.{0}d.txt'.format(embedding_dim))
embedding_matrix = get_embedding_matrix(word_index, embedding_type, embedding_path)

print 'Loading model...'
seq_length = 25
model_val = vqa_model(image_model_name, embedding_matrix, seq_length, dropout_rate=0.5, num_classes=3131)
model_val.load_weights(weights_file)

@app.route('/', methods=['GET', 'POST'])
def vqa_home():
    if request.method == 'POST':

        image_id = str(request.form['image_id'])
        image_name = 'abstract_v002_val2015_{0:012d}.png'.format(int(image_id))

        # if not in static copy it over
        if not image_cached(image_name):
            cache_image(image_name)

        # ask question
        xque = str(request.form['question'])

        if xque and image_id:
            xque_seq = data_loader.tokenizer.texts_to_sequences([xque])
            xque_seq = pad_sequences(xque_seq, maxlen=MAX_SEQUENCE_LENGTH)
            img_path = os.path.join(root, 'vqasite', 'static', 'img', image_name)
            ximg = load_image(img_path, (224, 224))
            pred = model_val.predict([np.array([ximg]),xque_seq])
            
            top = 5
            top_ans_ids = np.argsort(pred[0])[-top:][::-1]
            top_ans_probs = np.sort(pred[0])[-top:][::-1]
            top_ans_texts = []
            for i in range(top):
                ans_id = top_ans_ids[i]
                top_ans_texts.append(data_loader.selected_ans_list[ans_id])

            return render_template('vqa_home.html',
                                    image_id=image_id,
                                    image_name=image_name,
                                    question=xque,
                                    top_ans_texts=top_ans_texts,
                                    top_ans_probs=top_ans_probs)

        else:
            return render_template('vqa_home.html',
                                    image_id=image_id,
                                    image_name=image_name,
                                    question=xque)

    else:
        return render_template('vqa_home.html', home=True)

def image_cached(image_name):

    img_path = os.path.join(root, 'vqasite', 'static', 'img', image_name)
    return os.path.exists(img_path)

def cache_image(image_name):

    src_path = os.path.join(root, 'data', 'scene_img_abstract_v002_val2015', image_name)
    dst_path = os.path.join(root, 'vqasite', 'static', 'img', image_name)

    shutil.copy(src_path, dst_path)

