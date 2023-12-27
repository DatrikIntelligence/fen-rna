from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Reshape, Input, Dropout
from tensorflow.keras.layers import BatchNormalization, Lambda, Conv2DTranspose, Add
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import constraints
import tensorflow.keras.backend as K
import numpy as np
import inspect
from scoring import *
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from typing import List
import logging

ACTIVATIONS = ['relu', tf.keras.layers.LeakyReLU(alpha=0.1), 'tanh']
KERNELS = [(3,3), (10, 1), (10, 5)]      
SCORERS = [loss_score, 'mean_squared_error']

def dec(value, dtype, kernels=None):
    if dtype == int:
        value = int(round(value))
    elif dtype == bool:
        value = False if round(value) == 0 else True
    elif dtype == float:
        value = round(float(value), 15)
    elif dtype == 'scorer':
        value = SCORERS[dec(value, int)]
    elif dtype == 'activation':
        value = ACTIVATIONS[dec(value, int)]
    elif dtype == 'kernel':
        value = kernels[dec(value, int)]
    elif dtype == 'rnn_cell':
        value = tf.keras.layers.LSTMCell if int(round(value)) == 0 else tf.keras.layers.GRUCell
    elif dtype == str:
        value = str(value)
    elif dtype == tuple:
        value = value
    else:
        raise Exception("dtype %s does not found for decoding value" % dtype)
    
    return value

logging.basicConfig(level=logging.INFO)


def create_cnn_model(input_shape, block_size=2, nblocks=2, l1=1e-5, l2=1e-4, 
                     kernel_size=0, dropout=0.5, lr=1e-3, fc1=256, fc2=128,
                     conv_activation=2, dense_activation=2, dilation_rate=1,
                     batch_normalization=1, scorer=1, kernels=[(3,3), (10, 1), (10, 5)]):
    
    block_size = int(round(block_size))
    nblocks = int(round(nblocks))
    scorer = SCORERS[int(round(scorer))]
    fc1 = int(round(fc1))
    fc2 = int(round(fc2))
    dilation_rate = int(round(dilation_rate))
    conv_activation = ACTIVATIONS[int(round(conv_activation))]
    dense_activation = ACTIVATIONS[int(round(dense_activation))]
    kernel_size = kernels[int(round(kernel_size))]
    batch_normalization = True if batch_normalization == 1 else False
    
    input_tensor = Input(input_shape)
    x = input_tensor
    for i, n_cnn in enumerate([block_size] * nblocks):
        for j in range(n_cnn):
            x = Conv2D(32*2**min(i, 2), kernel_size=kernel_size, padding='same', 
                       kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                       kernel_initializer='he_uniform',
                       dilation_rate=dilation_rate)(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            x = Activation(conv_activation)(x)
        
        kernel = (min(2, x.shape[1]),min(2, x.shape[2]))

        x = MaxPooling2D(kernel)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

    x = Flatten()(x)
    
    # FNN
    x = Dense(fc1, 
             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = Activation(dense_activation)(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Dense(fc2, 
             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = Activation(dense_activation)(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Dense(1, activation='relu', name='predictions')(x) 
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss=scorer, optimizer=Adam(lr=lr), 
                  metrics=[NASAScore(), PHM21Score(), tf.keras.metrics.MeanAbsoluteError(name="MAE")])
    
    return model

class SelfAttention(tf.keras.layers.Layer):
     # Input shape 3D tensor with shape: `(samples, steps, features)`.
     # Output shape 2D tensor with shape: `(samples, features)`.

    def __init__(self, step_dim,W_regulizer = None,b_regulizer = None,
                 W_constraint = None, b_constraint = None, bias=True,**kwargs):
        
        self.W_regulizer = W_regulizer
        self.b_regulizer = b_regulizer
        
        self.W_constraint = W_constraint
        self.b_constraint = b_constraint
        self.bias = bias
        
        self.step_dim = step_dim
        self.features_dim = 0
        self.init = initializers.get('glorot_uniform')
        super(SelfAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {"W_regulizer": regularizers.serialize(self.W_regulizer),
                "b_regulizer": regularizers.serialize(self.b_regulizer),
                'b_constraint': constraints.serialize(self.b_constraint),
                'W_constraint': constraints.serialize(self.W_constraint),
                'bias': self.bias,
                'step_dim': self.step_dim
               }
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        assert len(input_shape) == 3
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(shape=(input_shape[-1],),
                                      initializer= self.init,
                                      constraint = self.W_constraint,
                                      regularizer = self.W_regulizer,
                                      name = '{}_W'.format(self.name))
        
        self.features_dim = input_shape[-1]
        

        self.b = self.add_weight(shape=(input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.b_regulizer,
                                 constraint=self.b_constraint)
    
        super(SelfAttention, self).build(input_shape)  

    
    def call(self, x, mask=None):
      
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
           
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def create_rnn_model(input_shape, bidirectional=0, attention=0, cell_type=0,
                     nblocks=2, rnn_units=64, l1=1e-5, l2=1e-4, dropout=0.5, lr=1e-3, 
                     fc1=256, fc2=128, dense_activation=2, batch_normalization=1, scorer=1):
    
    # params
    bidirectional = dec(bidirectional, bool)
    attention = dec(attention, bool)
    cell_type = dec(cell_type, 'rnn_cell')
    nblocks = dec(nblocks, int)
    rnn_units = dec(rnn_units, int)
    fc1 = dec(fc1, int)
    fc2 = dec(fc2, int)
    dense_activation = dec(dense_activation, 'activation')
    scorer = dec(scorer, 'scorer') 
    batch_normalization = dec(batch_normalization, bool)
    
    # model creation
    input_tensor = Input(input_shape)
    x = input_tensor

    rnn_units = [rnn_units] * nblocks
    for i, units in enumerate(rnn_units):
        #return_sequences = (i < len(rnn_units) - 1) or attention
        return_sequences = True
        cell = tf.keras.layers.RNN(cell_type(units=units, 
                                             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
                                   name='rnn_cell_%d' % (i+1),
                                   return_sequences=return_sequences,
                         )
        
        if bidirectional:
            x = tf.keras.layers.Bidirectional(cell)(x)
        else:
            x = cell(x)
            
        if batch_normalization:
            x = BatchNormalization()(x)
            
        
        if dropout > 0:
            x = Dropout(dropout)(x)

    if attention:
        x = SelfAttention(input_shape[0])(x)    
        x = Dropout(dropout)(x)
    
    # FNN
    x = Flatten()(x)
    x = Dense(fc1, 
             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = Activation(dense_activation)(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Dense(fc2, 
             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = Activation(dense_activation)(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Dense(1, activation='relu', name='predictions')(x) 
    model = Model(inputs=input_tensor, outputs=x)
    
    model.compile(loss=scorer, optimizer=Adam(lr=lr), 
                  metrics=[NASAScore(), PHM21Score(), tf.keras.metrics.MeanAbsoluteError(name="MAE")])
   
    return model


def create_mscnn_model(input_shape, block_size=2, nblocks=2, kernel_size=1, l1=1e-5, l2=1e-4, msblocks=2,
                       f1=10, f2=15, f3=20, dropout=0.5, lr=1e-3, filters=64,
                       fc1=256, fc2=128, conv_activation=2, dense_activation=2, 
                       dilation_rate=1, batch_normalization=1, scorer=1,  
                       kernels=[(3,3), (10, 1), (10, 5)]):
    
    block_size = dec(block_size, int)
    scorer = dec(scorer, 'scorer') 
    nblocks = dec(nblocks, int)
    msblocks = dec(msblocks, int)
    fc1 = dec(fc1, int)
    fc2 = dec(fc2, int)
    
    f1 = dec(f1, int)
    f2 = dec(f2, int)
    f3 = dec(f3, int)
    ms_kernel_size = [f1, f2, f3]
    
    dilation_rate = dec(dilation_rate, int)
    conv_activation = dec(conv_activation, 'activation')
    dense_activation = dec(dense_activation, 'activation')
    kernel_size = dec(kernel_size, 'kernel', kernels=kernels)
    batch_normalization = dec(batch_normalization, bool)

    input_tensor = Input(input_shape)
    x = input_tensor
    for i, _ in enumerate(range(msblocks)):

        cblock = []
        for k in range(3):
            output_shape = x.shape
            f = ms_kernel_size[k]
  
            b = Conv2D(filters, kernel_size=(f, 1), padding='same', 
                       kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                       kernel_initializer='he_uniform',
                       name='MSConv_%d%d_%d' % (i, k, f),
                       dilation_rate=dilation_rate)(x)

            if batch_normalization:
                b = BatchNormalization()(b)
            b = Activation(conv_activation)(b)

            cblock.append(b)

        x = Add()(cblock)
        if dropout > 0:
            x = Dropout(dropout)(x)
    
    
    for i, n_cnn in enumerate([block_size] * nblocks):
        for j in range(n_cnn):
            x = Conv2D(filters*2**min(i, 2), kernel_size=kernel_size, padding='same', 
                       kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                       kernel_initializer='he_uniform',
                       dilation_rate=dilation_rate)(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            x = Activation(conv_activation)(x)

        kernel = (min(2, x.shape[1]),min(2, x.shape[2]))
            
        x = MaxPooling2D(kernel)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)           

    x = Flatten()(x)
    
    # FNN
    x = Flatten()(x)
    x = Dense(fc1, 
             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = Activation(dense_activation)(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Dense(fc2, tf.keras.layers.LeakyReLU(alpha=0.1),
             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = Activation(dense_activation)(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Dense(1, activation='relu', name='predictions')(x) 
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss=scorer, optimizer=Adam(lr=lr), 
                  metrics=[NASAScore(), PHM21Score(), tf.keras.metrics.MeanAbsoluteError(name="MAE")])
    
    return model


def create_embeddings():

    for seed in [999, 666, 128, 256, 394]:

        train_gen = pk.load(open('data_set/train_%d.pk' % seed, 'rb'))
        test_gen = pk.load(open('data_set/test_%d.pk' % seed, 'rb'))

        train_gen.add_extra_channel = True
        train_gen.return_label = False
        train_gen.batch_size=256
        train_gen.epoch_len_reducer = 100
        test_gen.add_extra_channel = True
        test_gen.return_label = False
        test_gen.batch_size=256
        test_gen.epoch_len_reducer = 100
        train_gen.window_size = 1000
        test_gen.window_size = 1000 

        from livelossplot import PlotLossesKerasTF
        from models import VariationalAutoencoder

        VAE = VariationalAutoencoder(
                eta = 1,
                alpha = 1/50,
                loss_type='rmse',
                input_dim = (18,1000,1)
                , encoder_conv_filters = [32,64,64,64, 64]
                , encoder_conv_kernel_size = [3,3,3,3,3]
                , encoder_conv_strides = [1,2,2,2,1]
                , decoder_conv_filters = [64, 64,64,32,1]
                , decoder_conv_kernel_size = [3,3,3,3,3]
                , decoder_conv_strides = [1,2,2,2,1]
                , z_dim = 100
                , activation = 'relu')  

        VAE.compile(optimizer=Adam(0.0005))
        VAE.fit(x=train_gen, 
                     validation_data=test_gen,
                     batch_size=32, shuffle=True, epochs=4,
                     callbacks=[PlotLossesKerasTF()])

        for gen in [train_gen, test_gen]:

            for key in gen._X.keys():

                i = 0
                d = gen._X[key]
                vectors = np.zeros((d.shape[0]-1000, 100))
                bag = []
                print(key, d.shape[0]-1000)

                if os.path.exists("embeddings/emb_%d_%d.pk"  % (seed, key) ):
                    continue

                for c in range(0, d.shape[0]-1000, 1):
                    bag.append(d[c: c+1000].T)

                    if len(bag) == 256:
                        v = VAE.encoder.predict(np.array(bag))
                        vectors[i:i+256] = v
                        i+=256
                        bag = []

                if len(bag) != 0:
                    v = VAE.encoder.predict(np.array(bag))
                    vectors[i:] = v
                    print(key, i+v.shape[0])
                else:
                    print(key, i)

                pk.dump(vectors, open("embeddings/emb_%d_%d.pk" % (seed, key), "wb"))

        del train_gen
        del test_gen
        del vectors
        del VAE
        gc.collect()
