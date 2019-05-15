# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.preprocessing import sequence

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from actionEncode import EncodeMap3
from keras import backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import *
BATCHSIZE=128
maxlen=300


from keras.engine.topology import Layer


class MySumPool(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        self.axis = 1
        super(MySumPool, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def call(self, x, mask=None):
        output_mask = K.not_equal(x[1], 0)
        x =x[0]
        mask = K.reshape(output_mask,[-1,maxlen,1])
        # mask = K.repeat(output_mask, x.shape[-1])
        # mask = tf.transpose(mask, [0,2,1])
        mask = K.cast(mask, K.floatx())
        x = tf.multiply(x,mask)
        # x = K.print_tensor(x)
        # return K.sum(x, axis=1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], maxlen,1)

def get_GameScore(L,player_list):
    game_score = '0-0'
    tmp = []
    for x in L:
        if 'Conversion' in x:
            if x.split(':')[0] in player_list:
                game_score = x.split(':')[1].split('#')[-2]
                tmp.append(game_score)
        else:
            tmp.append('')
    return tmp

def getdata(size=-1):
    data = open('dataset/ball/2019-03-02.txt', 'r').read().split('\n')[:size]
    # winflag = np.array([[x.split('@')[0]] for x in data])
    players = np.array([[int(y.split(':')[0])+1 for y in x.split('@')[1].split(',')] for x in data])
    players_neg = np.array([[6-int(y.split(':')[0]) for y in x.split('@')[1].split(',')] for x in data])
    raw_actions = np.array([[y.split(':')[0]+':'+'#'.join(y.split(':')[1].split('#')[:3]) for y in x.split('@')[1].split(',')] for x in data])
    game_scores = np.array([get_GameScore(x.split('@')[1].split(','),['0','1','2']) for x in data])
    game_scores_neg = np.array([get_GameScore(x.split('@')[1].split(','),['3','4','5']) for x in data])
    actions = np.array([[EncodeMap3['#'.join(y.split(':')[1].split('#')[:3])] for y in x.split('@')[1].split(',')] for x in data])
    actions_neg = actions
    players = np.concatenate([players,players_neg])
    actions = np.concatenate([actions,actions_neg])
    game_scores = np.concatenate([game_scores,game_scores_neg])
    raw_actions = np.concatenate([raw_actions,raw_actions])
    players = sequence.pad_sequences(players, maxlen=maxlen,padding='post')
    actions = sequence.pad_sequences(actions, maxlen=maxlen,padding='post')
    y_train = np.array([1]*len(data)+[0]*len(data))
    return [players,actions],np.tile(y_train[:,np.newaxis],[1,maxlen]),(raw_actions,game_scores)

def mysum(dense):
    return K.sum(dense, axis=1)

def myaverage(dense):
    return tf.reduce_mean(dense, axis=1)

def aucw(dense):
    dense = K.cumsum(dense,axis=1)+0.5
    #clip or loss
    # dense = K.clip(dense,0,1)
    return tf.squeeze(dense,-1)
    # return K.sum(dense*tf.cast((tf.range(1,maxlen+1)[::-1]+1)/tf.cast(tf.to_float(100),tf.float32),tf.float32),axis=1)-tf.cast(K.constant(1*maxlen),tf.float32)

def mean_squared_error2(y_true, y_pred):
    return K.mean(K.abs(y_pred-y_true) + 50*(y_pred-1)*K.cast(K.greater(y_pred-1,0), K.floatx()) + 50*(0-y_pred)*K.cast(K.greater(0-y_pred,0), K.floatx()) ,axis=-1)

def train():
    # OUTPUT_UNIT = hp.output_unit
    # max_features = hp.vocab_size

    players_inputs = Input(shape=(maxlen,), dtype='int32')
    actions_inputs = Input(shape=(maxlen,), dtype='int32')
    embeddings_2 = Embedding(7, 16,mask_zero=True)(players_inputs)
    embeddings_3 = Embedding(len(EncodeMap3)+1, 128-16,mask_zero=True)(actions_inputs)
    embeddings = Concatenate(axis=-1)([embeddings_2,embeddings_3])
    input_lstm = LSTM(units=128, return_sequences=True)(embeddings)
    dense = TimeDistributed(Dense(1, activation='linear'))(input_lstm)
    # dense = TimeDistributed(Dense(1, activation='sigmoid'))(dense)

    output = MySumPool()([dense,players_inputs])
    output = Lambda(aucw)(output)
    # output = Lambda(myaverage)(output)

    # output = Lambda(mysum)(output)

    # output = K.sum(dense, axis=1)
    model = Model(input=[players_inputs,actions_inputs], output=[output])
    # if os.path.exists("model/lstm1.1.ALL.1.32.64.64.weights.014-0.9754.hdf5"):
    #     model_final.load_weights("model/lstm1.1.ALL.1.32.64.64.weights.014-0.9754.hdf5")
    model.compile(loss=mean_squared_error2,
                  optimizer='adam',
                  metrics=['mse','mae'])
    print(model.summary())

    [players,actions],y_train,_ = getdata()
    print('-------------------------')
    model.fit([players,actions], y_train,epochs=10,batch_size=BATCHSIZE,validation_data=([players[:100], actions[:100]], y_train[:100]))
    model.save('baseline.h5')


if __name__ == '__main__':
    train()


