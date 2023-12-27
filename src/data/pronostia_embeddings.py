import tensorflow as tf
import pandas as pd
import numpy as np
import random 
import logging
from data import pronostia
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import gc
import pickle as pk
import os
import models

FEATURE_NAMES = pronostia.FEATURE_NAMES

FOLD_TRAIN_BEARINGS = pronostia.FOLD_TRAIN_BEARINGS


def load_generators(train_config, test_config, return_test=True, return_train=True, fold=0):
    X_train, X_test = pronostia.load_data(return_test, return_train, 
                                train_bearings=FOLD_TRAIN_BEARINGS[fold])

    gen_train, gen_test = None, None
    
    if return_train:
        logging.info("Creating train generator")
        gen_train = EmbeddingsGenerator(X_train,  **train_config) 
        
        del X_train
        gc.collect()

    if return_test:
        logging.info("Creating test generator")
        gen_test = EmbeddingsGenerator(X_test, **test_config) 
        del X_test
        gc.collect()
    
    return gen_train, gen_test
    
compute_baseline = pronostia.compute_baseline

class EmbeddingsGenerator(tf.keras.utils.Sequence):
    """
    """
    def __init__(self, data, embedding_model_path, embedding_layer, batches_per_epoch=1000, batch_size=32, 
                 extra_channel=False, num_embeddings = 100, store_embeddings=True, 
                 step=100, channels=1):
        
        
        self.data = data
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.bearings = self.data[['Condition', 'Bearing']].drop_duplicates().values
        self.channels = channels
        
        self.num_embeddings = num_embeddings
        self.step = step 
        self.store_embeddings = store_embeddings
                                
        model = tf.keras.models.load_model(embedding_model_path, compile=False, 
                                                     custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU,
                                                                     'SelfAttention': models.SelfAttention})
        self._emb_model = tf.keras.models.Model(inputs=model.inputs, 
                                                  outputs=model.layers[embedding_layer].output)
        
        self.embedding_size = self._emb_model.output.shape[1]
        
        self._input_size = self._emb_model.input.shape[2]
        
        bearings = str(self.bearings).replace('\n', ',')
        logging.info(f"Pronostia (condition, bearing): {bearings}. Total bearings: {self.bearings.shape[0]}")
        
        self.data = {}
        self.rul_max = {}
        self.extra_channel = extra_channel
        self.__num_samples = data.shape[0]
        
        D = data
        for cond, bearing in self.bearings:
            d = D[(D.Condition == cond) & (D.Bearing==bearing)]
            self.rul_max[(cond, bearing)] = d.RUL.max()
            self.data[(cond, bearing)] = d[['V_acc', 'H_acc', 'RUL']].values
            
        self.emb = self._save_embeddings()
        
        for cond, bearing in self.bearings:
            rul = D[(D.Condition == cond) & (D.Bearing==bearing)].RUL.values
            self.data[(cond, bearing)] = (self._read_embeddings(bearing, cond), rul)
            
    def _save_embeddings(self):
        
        if not os.path.exists(self.store_embeddings):
            os.makedirs(self.store_embeddings)
        
        for cond, bearing in self.bearings:
            emb_file = os.path.join(self.store_embeddings, f"{cond}_{bearing}.pk")
            
            if not os.path.exists(emb_file):
                inputs = []
                data = self.data[(cond, bearing)]
                for i in range( (data.shape[0] - self._input_size)  // self.step):
                    pos = i * self.step
                    inputs.append((bearing, cond, pos))

                logging.info(f"Creating embeddings for bearing {bearing} and condition {cond}")
                X = self._get_embeddings(inputs)
                pk.dump(X, open(emb_file, "wb"))
                logging.info(f"Finished {bearing}  {cond}")
    
    def _read_embeddings(self, bearing, cond):
        emb_file = os.path.join(self.store_embeddings, f"{cond}_{bearing}.pk")
        return pk.load(open(emb_file, 'rb'))
            
    def _get_embeddings(self, inputs):
        
        X = np.zeros(shape=(len(inputs), 2,  self._input_size))
        for i, (bearing, cond, pos) in enumerate(inputs):
            X[i] = self.data[(cond, bearing)][pos:pos + self._input_size,  :2].T

        return self._emb_model.predict(X)
            
    def __len__(self):
        return self.batches_per_epoch
    
    
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        D = self.data
        
        total_embeddings = self.num_embeddings * self.channels
        X = np.zeros(shape=(self.batch_size, total_embeddings, self.embedding_size))
        Y = np.zeros(shape=(self.batch_size,))
        
        for i in range(self.batch_size):
            cond, bearing = self.bearings[random.randint(0, self.bearings.shape[0]-1)]
            emb, rul = self.data[(cond, bearing)]
            
           
            extra = (total_embeddings * (self._input_size - self.step)) // self.step
            pos = random.randint(0, len(emb) - total_embeddings - extra) 

            X[i] = emb[pos:pos + total_embeddings]
            
            ini = pos * self.step
            Y[i] = rul[ini + (total_embeddings * self._input_size)] / self.rul_max[(cond, bearing)]
        
        if self.extra_channel and self.channels == 1:
            X = np.expand_dims(X, axis=-1)
        elif self.channels >= 2:
            X = np.reshape(X, (self.batch_size, self.num_embeddings, self.embedding_size, self.channels))
        
        return X, Y
        
    def on_epoch_end(self):
        pass   
    