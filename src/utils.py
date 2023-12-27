import os
import h5py
import time
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np 
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Reshape, Input, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import pickle as pk
import gc
import ray
import multiprocessing
from ray import tune
from data import *
from ray import tune
import sys
import logging
import traceback
import pandas as pd
import h5py
from scoring import loss_score
from models import dec
from ray.tune.utils.util import unflatten_dict
from filelock import FileLock
from hashlib import sha1
from scoring import *
from typing import List
import inspect
import data
import importlib
import time



logging.basicConfig(level=logging.INFO)

CONFIG_DEC = {
    "arch_hash": (str, None),
    "net_hash": (str, None),
    "ae_hash": (str, None),
    "batch_size": (int, 32),
    "window": (int, 100),
    "l1": (float, 1e-3),
    "l2": (float, 1e-3),
    "dropout": (float, 0.9),
    "batch_normalization": (bool, True),
    "lr": (float, 1e-3),
    
    "variational_mode": (bool, True),
    "dec_blocks": (int, 4),
    "dec_activation" : ('activation', 'relu'),
    "eta": (float, 0.5),

    "block_size": (int, 4),
    "nblocks": (int, 4),
    "msblocks": (int, 4),

    "conv_activation": ('activation', 'relu'),
    "kernel_size": ('kernel', (3,3)),
    "filters" : (int, 32),
    "dilation_rate": (int, 1),
    
    "dense_activation": ('activation', 'relu'),
    "fc1": (int, 64),
    "fc2": (int, 100),
    "f1": (int, 5),
    "f2": (int, 10),
    "f3": (int, 15),
    "pretrain": (bool, True),
    "freeze": (bool, False),
    
    "ae_loss": (float, None),
    "ae_rec_loss": (float, None),
    "ae_kl_loss": (float, None),
    "net_score": (float, None),
    "net_mae": (float, None),
    "net_mse": (float, None),
    "net_NASA_score": (float, None),
    "status": (str, None),
    "ae_time": (float, None),
    "net_time": (float, None),
    
 
}


RNN_CONFIG_DEC = {
    "arch_hash": (str, None),
    "net_hash": (str, None),
    "ae_hash": (str, None),
    "batch_size": (int, 32),
    "window": (int, 100),
    "l1": (float, 1e-3),
    "l2": (float, 1e-3),
    "dropout": (float, 0.9),
    "lr": (float, 1e-3),
    "batch_normalization": (bool, True),
    
    "variational_mode": (bool, True),
    "dec_blocks": (int, 4),
    "dec_activation" : ('activation', 'relu'),
    "eta": (float, 0.5),

    "nblocks": (int, 4),

   
    "dense_activation": ('activation', 'relu'),
    "fc1": (int, 64),
    "fc2": (int, 100),
    "pretrain": (bool, True),
    "freeze": (bool, False),
    
    "ae_loss": (float, None),
    "ae_rec_loss": (float, None),
    "ae_kl_loss": (float, None),
    "net_score": (float, None),
    "net_mae": (float, None),
    "net_mse": (float, None),
    "net_NASA_score": (float, None),
    "status": (str, None),
    "ae_time": (float, None),
    "net_time": (float, None),
    
    "rnn_units": (int, None),
    "bidirectional": (bool, False),
    "attention": (bool, False),    
    "cell_type": ('rnn_cell', tf.keras.layers.LSTMCell),
    "rnn_units": (int, 32),    
    

}

CNNEMB_CONFIG_DEC = {
    "arch_hash": (str, None),
    "net_hash": (str, None),

    "batch_size": (int, 32),
    "window": (int, 100),
    "l1": (float, 1e-3),
    "l2": (float, 1e-3),
    "dropout": (float, 0.9),
    "batch_normalization": (bool, True),
    "lr": (float, 1e-3),
    
    "block_size": (int, 4),
    "nblocks": (int, 4),
    "msblocks": (int, 4),

    "conv_activation": ('activation', 'relu'),
    "kernel_size": ('kernel', (3,3)),
    "filters" : (int, 32),
    "dilation_rate": (int, 1),
    
    "dense_activation": ('activation', 'relu'),
    "fc1": (int, 64),
    "fc2": (int, 100),
    "f1": (int, 5),
    "f2": (int, 10),
    "f3": (int, 15),
    
    "net_score": (float, None),
    "net_mae": (float, None),
    "net_mse": (float, None),
    "net_NASA_score": (float, None),
    "status": (str, None),
    "net_time": (float, None),
    
    "step_min": (int, 100),
    "step_range": (int, 100),
    "channels": (int, 3),
    
}



def decconf(config, key, dtype, default=None, pop=True, kernels=None):
    if key not in config:
        return default
    
    if pop:
        value = config.pop(key)
    else:
        value = config[key]
    
    value = dec(value, dtype, kernels=kernels)
    
    return value

def decconfull(config, defaults=CONFIG_DEC):
    if 'kernels' in config:
        kernels = config['kernels']
    else:
        kernels = [(3,3)]
        
    for key, (dtype, default) in defaults.items():
        
        config[key] = decconf(config, key, dtype, default=default, pop=False, kernels=kernels)
        
        if hasattr(config[key], '__dict__'):         # is a user defined class
            config[key] = type(config[key]).__name__
        
    return config
        

def confighash(config, exclude=[]):
    if exclude is not None and len(exclude) > 0:
        config = config.copy()
        for key in exclude:
            if key in config:
                del config[key]
    
    return sha1(repr(sorted(config.items())).encode()).hexdigest()
        

def log_train(config):
    with FileLock('log_train.lock') as lock:
        try:
            log_path = 'log_train.csv'
            if os.path.exists(log_path):
                log = pd.read_csv(log_path)
                log = log.append(config, ignore_index=True)
            else:
                log = pd.DataFrame(data=[config])

            logging.info("Saving log train csv")
            log.to_csv(log_path, index=False)
        finally:
            lock.release()

def valid_params(config, func):
    config = config.copy()
    params = inspect.getargspec(func).args
    config = {k:v for k,v in config.items() if k in params}
    return config
            
HASH_EXCLUDE = ["ae_loss", "ae_rec_loss", "ae_kl_loss", "net_score", "net_mae", "net_mse", "net_NASA_score", 
                "net_hash", "arch_hash", "ae_hash", "status", "ae_time", "net_time"]
    

    
def format_values(_dict, values):
    for key, value in _dict.items():
        if isinstance(value, str):
            _dict[key] = value.format(**values)
    
def train(model_creator, config, ifold, dataset_config, queue, debug, verbose):
    logging.info('Starting training (fold %d) %s' % (ifold, config))
    time.sleep(random.randint(20,60))
    try:
        
        format_values(dataset_config['train_config'], locals())
        format_values(dataset_config['test_config'], locals())
        
        net_name = config.pop('net')
        experiment = config.pop('experiment')
        dec_defaults = CONFIG_DEC
        if 'dec_defaults' in config:
            thismodule = sys.modules[__name__]
            dec_defaults = getattr(thismodule, config.pop('dec_defaults'))

        csv_config = config.copy()
        csv_config['fold'] = ifold
        csv_config['net_name'] = net_name
        csv_config['experiment'] = experiment
        csv_config = decconfull(csv_config, defaults=dec_defaults)
        nhash = confighash(csv_config, exclude=HASH_EXCLUDE)
        csv_config['net_hash'] = nhash
        csv_config['arch_hash'] = confighash(csv_config, exclude=HASH_EXCLUDE + ["fold"])
        
        arch_dir =  "../temp/" + csv_config['arch_hash']
        if not os.path.exists(arch_dir):
            os.makedirs(arch_dir)
        
        net_path = f"{arch_dir}/net_{experiment}_{dataset_config['name']}_fold_{ifold}.h5"
        net_history = f"{arch_dir}/net_{experiment}_{dataset_config['name']}_fold_{ifold}_history.pk"
        
        if os.path.exists(net_path) and os.path.exists(net_history):
            logging.info(f"Found {net_path}")
            history = pk.load(open(net_history, 'rb'))
            queue.put(history)
            return
            
        config = config.copy()
                    
        # data reading
        logging.info("Reading data")
        W = decconf(config, "window", int) 
        
        
        logging.info(dataset_config)
        data = importlib.import_module(dataset_config['package'])
        train_gen, val_gen = data.load_generators(return_test=True, fold = ifold, 
                                                  train_config = dataset_config['train_config'],
                                                  test_config = dataset_config['test_config'])

        features= data.FEATURE_NAMES
        
        extra_channel = decconf(config, "extra_channel", bool, default=True)
        channels = decconf(config, "channels", int, default=1)
        num_embeddings = decconf(config, "num_embeddings", int, default=100)
        
        train_gen.channels = channels
        train_gen.extra_channel = extra_channel
        train_gen.num_embeddings = num_embeddings
        val_gen.channels = channels
        val_gen.extra_channel = extra_channel
        val_gen.num_embeddings = num_embeddings
        
        #for key, value in dataset_config["train_config"].items():
        #    setattr(train_gen, key, value)
        #for key, value in dataset_config["val_config"].items():
        #    setattr(val_gen, key, value)

        logging.info("Data read")
        
        
        # net config
        dec_blocks = decconf(config, "dec_blocks", int)  
        filters = decconf(config, "filters", int, pop=False)
        kernel_size = decconf(config, "kernel_size", float, pop=False) 
        dropout =  decconf(config, "dropout", float, pop=False)
        activation = decconf(config, "dec_activation", float)
        batch_normalization = decconf(config, "batch_normalization", float, pop=False)
        
        # define input shape
        input_shape = tuple(train_gen[0][0].shape[1:])

        
        # training config    
        epochs = decconf(config, "epochs", int)
        epochs = min(5, epochs) if debug else epochs
        batch_size = decconf(config, "batch_size", int)
        lr = decconf(config, "lr", float, pop=False)
        
        #
        train_epoch_len_reducer, test_epoch_len_reducer = decconf(config, "epoch_reducers", tuple, (500, 100))
        monitor = decconf(config, "monitor", str, "val_Score")
        
        
        model = model_creator(input_shape, **valid_params(config, model_creator))
        logging.info("Model created")
        
        model.summary(print_fn=lambda x: logging.info(x))
        
        
        es = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=8)
        rlr = tf.keras.callbacks.ReduceLROnPlateau(patience=3)
        
        
        logging.info("Started training")
        start_time = time.time()
        history = model.fit(train_gen, validation_data=val_gen, 
                            epochs=epochs, verbose=(2 if verbose else 0), 
                            callbacks=[es, rlr])
        history = history.history 

        csv_config['net_time'] = (time.time() - start_time)
        csv_config['net_score'] = history['val_Score'][-1]
        csv_config['net_mse'] = history['val_loss'][-1]
        csv_config['net_mae'] = history['val_MAE'][-1]
        csv_config['net_NASA_score'] = history['val_NASA_score'][-1]
        csv_config["status"] = "FINISHED"
        
        pk.dump(history, open(net_history, 'wb'))
        model.save(net_path)

        queue.put(history)
        
        log_train(csv_config)
                           
    except Exception as ex:
        if 'OOM' in str(ex):
            csv_config["status"] = "OOM ERROR"
        else:
            csv_config["status"] = "ERROR: " + str(ex)
            
        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)
        
        log_train(csv_config)
        

def train_with_embeddings(model_creator, config, seed, queue, debug, verbose, emb_file_name="embeddings/emb_%d_full.h5"):
    logging.info('Starting training (%d) %s' % (seed, config))
    try:
        dec_defaults = CNNEMB_CONFIG_DEC
        if 'dec_defaults' in config:
            thismodule = sys.modules[__name__]
            dec_defaults = getattr(thismodule, config.pop('dec_defaults'))

        csv_config = config.copy()
        csv_config['seed'] = seed
        csv_config = decconfull(csv_config, defaults=dec_defaults)
        nhash = confighash(csv_config, exclude=HASH_EXCLUDE)
        csv_config['net_hash'] = nhash
        csv_config['arch_hash'] = confighash(csv_config, exclude=HASH_EXCLUDE + ["seed"])
        
        net_path = '../temp/net_emb_%s.h5' % nhash
        
        net_history = '../temp/net_emb_%s_history.pk' % nhash
        if os.path.exists(net_path) and os.path.exists(net_history):
            history = pk.load(open(net_history, 'rb'))
            queue.put(history)
            return

        config = config.copy()
        epochs = decconf(config, 'epochs', int)
        epochs = min(5, epochs) if debug else epochs
        
        channels = decconf(config, 'channels', int) 
        step_min = decconf(config, 'step_min', int)  
        step_range = min(1000-step_min, decconf(config, 'step_range', int)) 
        step = (step_min, step_min + step_range)
        batch_size =  decconf(config, 'batch_size', int)  
        
        transpose_input = decconf(config, 'transpose_input', bool, default=False)
        input_embedding_size =  decconf(config, 'input_embedding_size', int)
        emb_file_name = decconf(config, 'emb_file_name', str, default=emb_file_name)
        train_epoch_len_reducer, test_epoch_len_reducer = decconf(config, "epoch_reducers", tuple, (2000, 400))

        # define input shape
        W = decconf(config, 'window', int)  

        if channels == 0 or not channels:
            if not transpose_input:
                input_shape = (W, 100)
            else:
                input_shape = (100, W)    
        elif channels > 0:
            input_shape = (100, W, channels)    
        else:
            raise Exception('channels value not valid: %d' % channels)

        from data import EmbeddingsGenerator
        try:
            file = emb_file_name % seed
            data, y = read_embedding(file, 'train')
            train_gen = EmbeddingsGenerator(data, y, seed, input_embedding_size=input_embedding_size)
 
            data, y = read_embedding(file, 'test')
            test_gen = EmbeddingsGenerator(data, y, seed, input_embedding_size=input_embedding_size)
            
        except Exception as ex:
            raise Exception('Error reading data', ex)
        
        train_gen.window = W
        train_gen.epoch_len_reducer = 8000 if debug else train_epoch_len_reducer
        train_gen.batch_size = batch_size
        train_gen.return_label = True
        train_gen.channels = channels
        train_gen.step = step
        train_gen.transpose_input = transpose_input
        
        test_gen.window = W
        test_gen.batch_size = 256
        test_gen.epoch_len_reducer = 4000 if debug else test_epoch_len_reducer
        test_gen.return_label = True
        test_gen.channels = channels
        test_gen.step = step
        test_gen.transpose_input = transpose_input
        

        model = model_creator(input_shape, **valid_params(config, model_creator))
        #logging.info(model.summary())
        #logging.info(train_gen.__getitem__(0)[0].shape)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_Score', patience=8)
        rlr = tf.keras.callbacks.ReduceLROnPlateau(patience=3)
        start_time = time.time()
        history = model.fit(train_gen, validation_data=test_gen, 
                            epochs=epochs, verbose=(2 if verbose else 0), 
                            callbacks=[es, rlr])
        history = history.history 
        
        csv_config['net_time'] = (time.time() - start_time)
        csv_config['net_score'] = history['val_Score'][-1]
        csv_config['net_mse'] = history['val_loss'][-1]
        csv_config['net_mae'] = history['val_MAE'][-1]
        csv_config['net_NASA_score'] = history['val_NASA_score'][-1]
        csv_config["status"] = "FINISHED"
        
        pk.dump(history, open(net_history, 'wb'))
        model.save(net_path)

        queue.put(history)
        
        log_train(csv_config)
                           
    except Exception as ex:
        if 'OOM' in str(ex):
            csv_config["status"] = "OOM ERROR"
        else:
            csv_config["status"] = "ERROR: " + str(ex)
            
        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)
        
        log_train(csv_config)
 
    
def parameter_opt_cv(model_creator, config, dataset_config, target=train):
    try:
        min_score = config.pop('min_score')
        monitor = decconf(config, "monitor", str, 'val_Score', pop=False) 
        timeout = decconf(config, "timeout", int)
        train_verbose = config.pop('train_verbose')
        
        debug = decconf(config, "debug", bool, False)
        if debug:
            pass

        wd = config.pop('working_dir')
        os.chdir(wd)

        data = config.copy()
        data['model'] = model_creator.__name__
        data['folds'] = {}

        # cross-validation
        finish = False
        for ifold in range(dataset_config['num_folds']):
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=target, args=(model_creator, config, ifold, dataset_config, 
                                                             queue, debug, train_verbose))
            p.start()
            p.join(timeout)
            if p.is_alive():
                logging.info('Fold %d timeout' % ifold)
                p.terminate()
                p.join()

                finish = True
            else:
                r = queue.get()
                if r is None:
                    finish = True

                else:
                    data['folds'][ifold] = r

            if len(data['folds'].keys()) > 0:
                # compute the mean score
                epochs = [len(data['folds'][ifold][monitor]) for ifold in data['folds'].keys() ]
                scores = [data['folds'][ifold][monitor][-1] for ifold in data['folds'].keys() ]

                if 'val_NASA_score' in data['folds'][ifold]:
                    nasa_scores = [data['folds'][ifold]['val_NASA_score'][-1] for ifold in data['folds'].keys() ]
                    mae_scores = [data['folds'][ifold]['val_MAE'][-1] for ifold in data['folds'].keys() ]

                    ray.train.report({"score":np.mean(scores), "mean_epochs":np.mean(epochs), 
                            "std_score":np.std(scores), "nasa_score":np.mean(nasa_scores),
                            "mae":np.mean(mae_scores)}
                           )
                else:
                    ray.train.report({"score":np.mean(scores), "mean_epochs":np.mean(epochs)})
                    
                    

                if np.mean(scores) > min_score:
                    logging.info("Stop train because mean scores %f > min score %f" % (np.mean(scores), min_score) )
                    logging.info("Scores: %s" % scores)
                    sys.stdout.flush()
                    return

            elif finish:
                logging.info("Not finished any trial")
                ray.train.report({"score":999, "mean_epochs":999, "std_score":999, "nasa_score":999, "mae":999})

            if finish:
                logging.info("Finished train")
                return


        # compute the mean score
        epochs = [len(data['folds'][ifold][monitor]) for ifold in data['folds'].keys() ]
        scores = [data['folds'][ifold][monitor][-1] for ifold in data['folds'].keys() ]
        
        if 'val_NASA_score' in data['folds'][ifold]:
            nasa_scores = [data['folds'][ifold]['val_NASA_score'][-1] for ifold in data['folds'].keys() ]
            mae_scores = [data['folds'][ifold]['val_MAE'][-1] for ifold in data['folds'].keys() ]

            logging.info("Scores: %s" % scores)
            sys.stdout.flush()
            ray.train.report({"score":np.mean(scores), "mean_epochs":np.mean(epochs), 
                            "std_score":np.std(scores), "nasa_score":np.mean(nasa_scores),
                            "mae":np.mean(mae_scores)}
                       )
                    
        else:
            ray.train.report({"score":np.mean(scores), "mean_epochs":np.mean(epochs)})
                    
                            
    except Exception as ex:
        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)
