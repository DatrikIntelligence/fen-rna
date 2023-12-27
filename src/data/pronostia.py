import tensorflow as tf
import pandas as pd
import numpy as np
import random 
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import gc

FEATURE_NAMES = ["V_acc", "H_acc"]

FOLD_TRAIN_BEARINGS = [
    [1, 3, 6, 7],
    [4, 2, 7, 5],
    [4, 3, 5, 1],
    [6, 2, 3, 5],
    [4, 7, 5, 1],
    [5, 2, 7, 4],
    [3, 4, 7, 2],
]

def load_data(return_test=True, return_train=True, train_bearings=[1, 3, 6, 7]):
    X_train = pd.read_csv('../data/pronostia/pronostia_train.csv')
    X_test = pd.read_csv('../data/pronostia/pronostia_test.csv')
    X = pd.concat((X_train,X_test), axis=0)
    print(X.Bearing.unique())
    
    scaler = StandardScaler()
    X.loc[:, FEATURE_NAMES] = scaler.fit_transform(X.loc[:,FEATURE_NAMES]).round(3)
    
    test_bearings = [i for i in range(1, 8) if i not in train_bearings]
    X_test = X[X.Bearing.isin(test_bearings)]
    X_train = X[X.Bearing.isin(train_bearings)]

    return X_train, X_test

def load_generators(return_test=True, return_train=True, tslen=128, fold=0):
    X_train, X_test = load_data(return_test, return_train, 
                                train_bearings=FOLD_TRAIN_BEARINGS[fold])

    gen_train, gen_test = None, None
    
    if return_train:
        logging.info("Creating train generator")
        gen_train = Sequence(X_train, 
                                extra_channel=False, batch_size=32, 
                                batches_per_epoch=10000, ts_len=tslen) 
        
        del X_train
        gc.collect()

    if return_test:
        logging.info("Creating test generator")
        gen_test = Sequence(X_test, 
                                extra_channel=False, batch_size=32, 
                                batches_per_epoch=5000, ts_len=tslen) 
        del X_test
        gc.collect()
    
    return gen_train, gen_test
    
def compute_baseline(Y_train, Y_test):
    y_mean, y_std = Y_train.mean(), Y_train.std()
    y_pred = [y_mean] * Y_test.shape[0]
    
    logging.info(f"Mean prediction: {y_mean}")

    mae = tf.keras.metrics.MeanAbsoluteError(name='mae')(Y_test, y_pred).numpy()
    mse = tf.keras.metrics.MeanSquaredError(name='mse')(Y_test, y_pred).numpy()
    
    logging.info(f"MAE: {mae}")
    logging.info(f"MSE: {mse}")
    
    return {'mae': mae, 'mse': mse}


# Data generator 
class Sequence(tf.keras.utils.Sequence):

    def __init__(self, data, batches_per_epoch=1000, batch_size=32, extra_channel=False, ts_len=256):
        self.data = data
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.bearings = self.data[['Condition', 'Bearing']].drop_duplicates().values
        
        bearings = str(self.bearings).replace('\n', ',')
        logging.info(f"Pronostia (condition, bearing): {bearings}. Total bearings: {self.bearings.shape[0]}")
        
        self.data = {}
        self.rul_max = {}
        self.extra_channel = extra_channel
        self.__num_samples = data.shape[0]
        self.ts_len = ts_len
        D = data
        for cond, bearing in self.bearings:
            d = D[(D.Condition == cond) & (D.Bearing==bearing)]
            self.rul_max[(cond, bearing)] = d.RUL.max()
            self.data[(cond, bearing)] = d[['V_acc', 'H_acc', 'RUL']].values
            
    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):
        D = self.data
        ts_len = self.ts_len

        if self.extra_channel:
            X = np.zeros(shape=(self.batch_size, 2, ts_len, 1))
        else:
            X = np.zeros(shape=(self.batch_size, 2, ts_len))

        Y = np.zeros(shape=(self.batch_size,))
        for i in range(self.batch_size):
            cond, bearing = self.bearings[random.randint(0, self.bearings.shape[0]-1)]
            Db = self.data[(cond, bearing)]
            L = (Db.shape[0] // ts_len) 
            k = random.randint(0, L-2) * ts_len
            

            if self.extra_channel:
                X[i, :, :, 0] = Db[k:k+ts_len, :2].T
            else:
                X[i, :, :] = Db[k:k+ts_len, :2].T
            Y[i] = Db[k:k+ts_len, 2][-1] / self.rul_max[(cond, bearing)]  
        
        return X, Y

class EmbeddingsGenerator(Sequence):
    """
    """
    def __init__(self, X, Y, batch_size=32, 
                 batches_per_epoch=1000, num_embeddings = 100, 
                 channels=1, step=100, embedding_size=100,
                 transpose_input=False, max_zeropad_ratio=0.1,
                 zeropad_prob=0.1):

        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self._ids = list(X.keys())
        self._Y = Y
        self._X = X
        self.__len = sum(len(Y[_id]) for _id in Y.keys())
        self.channels = channels
        self.num_embeddings = num_embeddings
        self.step = step 
        self.embedding_size = embedding_size
        self.transpose_input = transpose_input
        self.max_zeropad_ratio = max_zeropad_ratio
        self.zeropad_prob = zeropad_prob
        

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(self.__len / self.epoch_len_reducer / self.batch_size)
    
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        step_range = self.step if isinstance(self.step, tuple) else (self.step, self.step+1)
        latent_size = 100
        ids = np.random.choice(self._ids, size=self.batch_size, replace=True)
        ids = Counter(sorted(ids))
        channels = self.channels if self.channels else 1
        encoders = np.zeros((self.batch_size*channels*self.window, latent_size))
        y = np.zeros((self.batch_size,))
        ei = 0
        si = 0
        for _id, n in ids.items():
            nrows = len(self._Y[_id])

            data = self._X[_id]
            
            for _ in range(n):
                jump = random.choice(list(range(*step_range)))
                
                try:
                    cut = random.randint(0, nrows - self.window * jump * channels - self.input_embedding_size)
                except Exception as ex:
                    print("ERROR AF5", nrows, self.window, jump, channels, self.input_embedding_size)
                    raise ex
                    
                
                for j in range(self.window * channels):
                    c = cut + (jump * j)
                    encoders[ei, :] = data[c].T
                    
                    ei += 1
                
                if random.uniform(0,1) < self.zeropad_prob:
                    max_padding = int(self.max_zeropad_ratio * latent_size)
                    zero_padding = random.choice(list(range(0, max_padding)))
                    encoders[:zero_padding, :] = 0 
                
                # label (RUL)
                y[si] = self._Y[_id][c + self.input_embedding_size]
                si += 1  
       
        _X = np.zeros((self.batch_size, self.window, latent_size, channels))
        nz = self.window * channels
        for i in range(self.batch_size): 
            _X[i] = encoders[i*nz:nz*(i+1),:].T.reshape(self.window, latent_size, channels)
            
        _X = _X.astype(np.float32)
        _y = np.array(y).reshape((self.batch_size, 1)).astype(np.float32)
        
        if not self.channels:
            _X = np.squeeze(_X, axis=-1)
        
        if self.transpose_input:
            _X = _X.transpose((0,2,1))        

        return _X, _y
        
    def on_epoch_end(self):
        pass   
    