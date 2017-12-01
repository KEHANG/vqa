#################################################
######   Defines combined VGG/LSTM model     #####
#####     and dense fused model for VQA   ########
##################################################

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Merge
from keras.applications.vgg16 import VGG16
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

###########################
## VGG model for image ##
###########################
def vgg_model():
    print "Creating VGG image model..."
    # Generate a model with all layers 
    vgg16 = VGG16(include_top=True)
    # Make vgg16 model layers as non-trainable (freeze)
    for layer in vgg16.layers:
        layer.trainable = False
    # Add a layer where input is the output of the second last layer (before softmax)
    new_layer = Dense(1024, input_dim=4096, activation='tanh')(vgg16.layers[-2].output)
    #Then create the corresponding model
    model = Model(inputs=vgg16.input, outputs=new_layer)
    return model

#########################
## Aggregate VQA model ##
#########################
def vqa_model(embedding_matrix, seq_length, dropout_rate, num_classes):
    vgg = vgg_model()
    lstm_model = Word2VecModel(embedding_matrix, seq_length, dropout_rate)
    print "Merging final model..."
    fc_model = Sequential()
    fc_model.add(Merge([vgg, lstm_model], mode='mul'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(1000, activation='tanh'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(num_classes, activation='softmax'))
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
    return fc_model

