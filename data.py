import os
import json
import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_data():

    question_train_file = 'data/Questions_Train_abstract_v002/MultipleChoice_abstract_v002_train2015_questions.json'

    question_train_dict = json.load(open(question_train_file))

    x_train_dict = {}
    for item in question_train_dict['questions']:
        image_id = item['image_id']
        question_id =  item['question_id']
        key = "{0}_{1}".format(image_id, question_id)
        x_train_dict[key] = item['question']

    question_val_file = 'data/Questions_Val_abstract_v002/MultipleChoice_abstract_v002_val2015_questions.json'

    question_val_dict = json.load(open(question_val_file))

    x_val_dict = {}
    for item in question_val_dict['questions']:
        image_id = item['image_id']
        question_id =  item['question_id']
        key = "{0}_{1}".format(image_id, question_id)
        x_val_dict[key] = item['question']

    ans_train_file = 'data/Questions_Train_abstract_v002/abstract_v002_train2015_annotations.json'

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

    ans_val_file = 'data/Questions_Val_abstract_v002/abstract_v002_val2015_annotations.json'

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

    y_train = keras.utils.to_categorical(y_train, len(selected_ans_list))

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

    y_val = keras.utils.to_categorical(y_val, len(selected_ans_list))

    return (ximgid_train, xque_train, y_train), (ximgid_val, xque_val, y_val)

def load_image(img_path, size=(224, 224)):

    img_obj = Image.open(img_path)
    img_obj = img_obj.resize(size)

    img_arry = np.array(img_obj)
    img_arry = img_arry[:,:,:3]
    
    return img_arry

