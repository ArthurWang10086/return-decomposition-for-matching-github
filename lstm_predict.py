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
from lstm import maxlen,MySumPool,getdata,mean_squared_error2

def get_outputs(input_data, model,layer_names):
    # layer_names = [layer.name for layer in model.layers if
    #                'flatten' not in layer.name and 'input' not in layer.name
    #                and 'predictions' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    layer_outputs = intermediate_layer_model.predict(input_data)
    return layer_outputs, layer_names




if __name__ == '__main__':
    from keras.models import load_model
    model = load_model('baseline.h5',custom_objects={'MySumPool':MySumPool,"tf": tf,"mean_squared_error2":mean_squared_error2})
    x_train,y_train,(raw_actions,game_scores) = getdata(20)
    exmaple_index = 6
    print([layer.name for layer in model.layers])
    layer_outputs = get_outputs(x_train, model,['my_sum_pool_1','lambda_1'])[0][1]
    # print(layer_outputs.shape)
    # print(len(raw_actions[0]),raw_actions[0])
    # print(layer_outputs)
    print(list(zip(raw_actions[exmaple_index],game_scores[exmaple_index],layer_outputs[exmaple_index].flatten())))
    # print(layer_outputs[0].flatten())
    # print(sum(layer_outputs[0].flatten()))
    # print(len(layer_outputs[0].flatten()))

    result = model.predict(x_train)
    print(np.mean(np.abs(np.array(result)-y_train),-1))

    # print(sum(layer_outputs[0].flatten()[:len(raw_actions[0])]))
    # print(sum(layer_outputs[0].flatten()[-len(raw_actions[0]):]))
    x=list(zip(raw_actions[exmaple_index],game_scores[exmaple_index],layer_outputs[exmaple_index].flatten()))
    # x=list(zip(raw_actions[0],np.cumsum(layer_outputs[0].flatten())))
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame(x)
    # print(df)
    # print(df.loc[:,1])
    fig = plt.figure( )
    ax = df.plot(kind='line')
    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.loc[:,1], rotation=90)
    # ax.set_title(col)
    plt.show()





