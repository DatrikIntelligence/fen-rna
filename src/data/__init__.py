import random
import numpy as np 
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from collections import Counter
from pyts.image import GramianAngularField
import h5py

class DataGenerator(Sequence):
    """
    """
    def __init__(self, X, attributes, window_size=10, batch_size=32, 
                 noise_level=0, epoch_len_reducer=100, add_extra_channel=False,
                 output_extra_channel=False, return_label=True):
        self.batch_size = batch_size
        self.return_label = return_label
        self.output_extra_channel = output_extra_channel
        self.window_size = window_size
        self.noise_level = noise_level
        self.attributes = attributes
        self.epoch_len_reducer = epoch_len_reducer
        self._X = {}
        self._Y = {}
        self._ids = X.id.unique()
        self.add_extra_channel = add_extra_channel
        for _id in self._ids:
            self._X[_id] = X.loc[(X.id==_id), self.attributes].values
            self._Y[_id] = X.loc[(X.id==_id), 'Y'].values
        self.__len = int((X.groupby('id').size() - self.window_size).sum() / 
                        self.batch_size)
        del X


    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(self.__len / self.epoch_len_reducer)
    
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        X = self._X
        _X = []
        _y = []
        for _ in range(self.batch_size):
            sid = random.choice(self._ids)
            unit = self._X[sid]
            nrows = unit.shape[0]
            cut = random.randint(0, nrows - self.window_size)
            s = unit[cut: cut + self.window_size].T
            y =self._Y[sid][cut + self.window_size-1]
            _X.append(s)
            _y.append(y)

        
        _X = np.array(_X)
        if self.add_extra_channel:
            _X = _X.reshape(_X.shape + (1,))
            
        if self.noise_level > 0:
            noise_level = self.noise_level
            noise = np.random.normal(-noise_level, noise_level, _X.shape)
            _X = _X + noise
            _X = (_X - _X.min()) / (_X.max() - _X.min())
       
        if self.return_label:
            return _X, np.array(_y).reshape((self.batch_size, 1))
        elif self.output_extra_channel:
            return _X, _X.reshape(_X.shape + (1,))
        else:
            return _X, _X
        
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        pass

    

class EmbeddingsGenerator(Sequence):
    """
    """
    def __init__(self, X, Y, batch_size=32, 
                 epoch_len_reducer=100, window_size = 100, 
                 channels=1, step=1000, input_embedding_size=1000,
                 transpose_input=False, max_zeropad_ratio=0.1,
                 zeropad_prob=0.1):
        self.batch_size = batch_size
        
        self.epoch_len_reducer = epoch_len_reducer
        self._ids = list(X.keys())
        self._Y = Y
        self._X = X
        self.__len = sum(len(Y[_id]) for _id in Y.keys())
        self.channels = channels
        self.window = window_size
        self.step = step 
        self.input_embedding_size = input_embedding_size
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
    
