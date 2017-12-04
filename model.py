#################################################
######   Defines combined VGG/LSTM model     #####
#####     and dense fused model for VQA   ########
##################################################

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Merge
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.models import Model

#######################
# LSTM model for text #
#######################
def Word2VecModel(embedding_matrix, seq_length, dropout_rate):
    print "Creating text model..."
    num_words, embedding_dim = embedding_matrix.shape
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, 
        weights=[embedding_matrix], input_length=seq_length, trainable=False))
    model.add(LSTM(units=512, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=512, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1024, activation='tanh'))
    return model

##############################
## CNN model for image ##
##############################
def image_model(model_name='vgg16'):
    print "Creating {0} image model...".format(model_name)
    # Generate a model with all layers
    if model_name == 'vgg16':
        pretrained_model = VGG16(include_top=True)
    elif model_name == 'vgg19':
        pretrained_model = VGG19(include_top=True)
    elif model_name == 'resnet50':
        pretrained_model = ResNet50(include_top=True)
    elif model_name == 'xception':
        pretrained_model = Xception(include_top=True)
    else:
        raise "Pre-trainied {0} not included yet.".format(model_name)
    # Make vgg16 model layers as non-trainable (freeze)
    for layer in pretrained_model.layers:
        layer.trainable = False
    # Add a layer where input is the output of the second last layer (before softmax)
    new_layer = Dense(1024, activation='tanh')(pretrained_model.layers[-2].output)
    #Then create the corresponding model
    model = Model(inputs=pretrained_model.input, outputs=new_layer)
    return model

#########################
## Aggregate VQA model ##
#########################
def vqa_model(image_model_name, embedding_matrix, seq_length, dropout_rate, num_classes):
    img_model = image_model(model_name=image_model_name)
    lstm_model = Word2VecModel(embedding_matrix, seq_length, dropout_rate)
    print "Merging final model..."
    fc_model = Sequential()
    fc_model.add(Merge([img_model, lstm_model], mode='mul'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(1000, activation='tanh'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(num_classes, activation='softmax'))
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
    return fc_model

