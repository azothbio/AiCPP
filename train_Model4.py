#%% import module
from tqdm import tqdm
from myCode import MCC, getJson

## Turn-off all warning messages
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

## tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kL
from tensorflow.keras import models as kM

## sklearn metrics
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, r2_score

import shutil as sh
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import argparse

init_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) # ~/git/CPP_2nd/
script_path = os.path.join(init_path, 'scripts_new') # ~/git/CPP_2nd/scripts_new
dataset_path = os.path.join(init_path, 'dataset_new') # ~/git/CPP_2nd/dataset_new
assert os.path.exists(script_path) and os.path.exists(dataset_path), f"There is no path! Check next two path\n{script_path}\n{dataset_path}"

#%% make Xn, Xc
def make_Xn_Xc(df_trX, df_prX):
    aachars="ACDEFGHIKLMNPQRSTVWY-"
    d_aa2i = {}
    for idx, aa in enumerate(aachars):
        d_aa2i[aa] = idx
    
    def get_Xn(d_aa2i, seq):
        xseq = [d_aa2i[aa] for aa in seq]
        Xn = np.array(xseq, dtype=np.int8)
        return Xn
    
    def get_Xc(Xn):
        return np.array(Xn[::-1], dtype=np.int8)

    df_trX.loc[:, 'Xn'] = df_trX['seq'].apply(lambda x: get_Xn(d_aa2i, x))
    df_trX.loc[:, 'Xc'] = df_trX['Xn'].apply(lambda x: get_Xc(x))
    
    df_prX.loc[:, 'Xn'] = df_prX['seq'].apply(lambda x: get_Xn(d_aa2i, x))
    df_prX.loc[:, 'Xc'] = df_prX['Xn'].apply(lambda x: get_Xc(x))

    return df_trX, df_prX

class XY_data:
    def __init__(self, df, aachars="ACDEFGHIKLMNPQRSTVWY-"):
        self.seq = list(df['seq'])
        self.Xn = np.array(list(df['Xn']))
        self.Xc = np.array(list(df['Xc']))
        self.sample_weight = np.array(list(df['weight']))
        self.y = np.array(list(df['y']))

#%% get df_trX, df_prX
def get_df_trprX(df_pos_neg_path, kFold_num, kFold_idx, use_save=False):
    
    assert os.path.exists(df_pos_neg_path), f"There is no path! Check next path\n{df_pos_neg_path}"
    os.makedirs(os.path.join(dataset_path, 'train'), exist_ok=True)
    df_trX_path = os.path.join(dataset_path, 'train', f'cpp_trX_{win_size}mer_{kFold_num}_{kFold_idx}.csv')
    df_prX_path = os.path.join(dataset_path, 'train', f'cpp_prX_{win_size}mer_{kFold_num}_{kFold_idx}.csv')

    # check [ df_trX.csv, df_prX.csv ] file
    if all([os.path.exists(df_trX_path), os.path.exists(df_prX_path)]):
        df_trX = pd.read_csv(df_trX_path, index_col=False)
        df_prX = pd.read_csv(df_prX_path, index_col=False)

    else:
        # split df_trX, df_prX by using KFold
        df_pos_neg = pd.read_csv(df_pos_neg_path)
        kf = StratifiedKFold(kFold_num, shuffle=True, random_state=101)
        result = [i for i in kf.split(df_pos_neg, df_pos_neg['y'])]
        
        df_trX = df_pos_neg.iloc[result[kFold_idx][0]]
        df_prX = df_pos_neg.iloc[result[kFold_idx][1]]
        df_prX = df_prX.drop_duplicates('seq')

    if use_save:
        df_trX.to_csv(df_trX_path)
        df_prX.to_csv(df_prX_path)
        
    return df_trX, df_prX
    
def get_trprX(dataset_path, kFold_num, kFold_idx, df_trX, df_prX, use_save=False):
    trX_path = os.path.join(dataset_path, 'train', f'trX_{win_size}mer_{kFold_num}_{kFold_idx}.pkl')
    prX_path = os.path.join(dataset_path, 'train', f'prX_{win_size}mer_{kFold_num}_{kFold_idx}.pkl')
    
    if all([os.path.exists(trX_path), os.path.exists(prX_path)]):
        with open(trX_path, 'rb') as f:
            trX = pickle.load(f)
        with open(prX_path, 'rb') as f:
            prX = pickle.load(f)

    else:
        df_trX, df_prX = make_Xn_Xc(df_trX, df_prX)    
        trX = XY_data(df_trX)
        prX = XY_data(df_prX)
    
    if use_save:
        with open(trX_path, 'wb') as f:
            pickle.dump(trX, f)
        with open(prX_path, 'wb') as f:
            pickle.dump(prX, f)
    
    return trX, prX


#%%
class myCallBack(tf.keras.callbacks.Callback):
    def __init__(self, model, trX, prX, File_prX_predict):
        self.model = model
        self.trX = trX
        self.prX = prX
        self.prX_pred = File_prX_predict
        
    def save_prd(self, epoch, y, fx):
        save_path = f"self.prX_pred_{epoch}"
        with open(save_path, 'w') as f:
            print("y,fx", file=f)
            for idx, _y in enumerate(y):
                print(f"{self.prX.seq},{_y:0.1f},{fx[idx]:0.3f}", file=f)

    def on_epoch_end(self, epoch, logs={}):
        self.model.metrics[1].print_confusion()


#%% Functional-API 
def myCnn(tag, x, nf=8, ks=3, st=2, act='swish', pad='valid'):
    x = kL.Conv1D(filters=nf, kernel_size=ks, strides=st, padding=pad, activation=act, name='%s_cnn' % tag)(x)
    return x


def resblock(tag, x, lat_dim):
    fx = myCnn(f"{tag}_1", x, lat_dim, ks=3, st=1, act=None, pad='same')
    fx = kL.BatchNormalization()(fx)
    fx = kL.Activation('swish')(fx)
    
    fx = myCnn(f"{tag}_2", fx, lat_dim, ks=3, st=1, act=None, pad='same')
    fx = kL.BatchNormalization()(fx)
    fx = kL.Add()([x,fx])
    out = kL.Activation('swish')(fx)
    
    return out


def identity_block(tag, x, nf=8, ks=3):
    x_shortcut=x
    x = kL.Conv1D(filters=nf, kernel_size=1, strides=1, padding='valid', name=f'{tag}_c1d_1a')(x)
    x = kL.BatchNormalization(name=f'{tag}_bn_1b')(x)
    x = kL.Activation('swish')(x)
    x = kL.Conv1D(filters=nf, kernel_size=ks, strides=1, padding='same', name=f'{tag}_c1d_2a')(x)
    x = kL.BatchNormalization(name=f'{tag}_bn_2b')(x)
    x = kL.Activation('swish')(x)
    x = kL.Conv1D(filters=nf, kernel_size=1, strides=1, padding='valid', name=f'{tag}_c1d_3a')(x)
    x = kL.BatchNormalization(name=f'{tag}_bn_3b')(x)
    x = kL.Add()([x,x_shortcut])
    x = kL.Activation('swish')(x)
    return x


def attN(tag, xV, xQ, dropout):
    xA = kL.Attention(name='%s_att' % tag, use_scale=True, dropout=dropout)([xV, xQ])
    xC = kL.Add(name='%s_con' % tag)([xQ, xA])
    return xA, xC


def myDense(inputX, nodeNum, activation='swish', dropout=0.1, use_name=False):
    if use_name:
        x = kL.Dense(nodeNum, name=use_name)(inputX)
    else:
        x = kL.Dense(nodeNum)(inputX)
    x = kL.Activation(activation)(x)
    x = kL.Dropout(dropout)(x)
    return x


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, position_size, token_dim, embedding_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(token_dim, embedding_dim)
        self.pos_emb = tf.keras.layers.Embedding(position_size, embedding_dim)
    def call(self, x):
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        token = self.token_emb(x)
        return token + positions


#%% model_structure
def init_model(trX, config):
    _DIM_LAT = 5
    _DROP_OUT = 0.3
    
    Xinp_n = kL.Input(shape=trX.Xn.shape[1:], name='Xinp_n')
    
    l_emb = kL.Embedding(input_dim=21, output_dim=3, name='l_emb')
    xemb = l_emb(Xinp_n)
    xA, xC = attN('att2', xemb, xemb, 0.3)

    xemb = kL.Flatten(name='flat_xemb')(xemb)
    xA = kL.Flatten(name='flat_xA')(xA)
    xC = kL.Flatten(name='flat_xC')(xC)

    xemb = kL.Dense(_DIM_LAT*2, activation='relu', name='d1_xemb')(xemb)
    xA = kL.Dense(_DIM_LAT*2, activation='relu', name='d1_xA')(xA)
    xC = kL.Dense(_DIM_LAT*2, activation='relu', name='d1_xC')(xC)

    x = kL.Concatenate(name='concat_xAC')([xemb, xA, xC])
    x = kL.Dropout(_DROP_OUT)(x)

    x = kL.Dense(_DIM_LAT*2, activation='relu', name='d1_x')(x)
    x = kL.Dropout(_DROP_OUT)(x)

    x = kL.Dense(_DIM_LAT, activation='relu', name='d2_x')(x)
    x = kL.Dropout(_DROP_OUT)(x)

    fx = kL.Dense(1, activation='sigmoid', name='fx')(x)

    model = kM.Model(inputs=Xinp_n, outputs=fx, name='model_cpp2')
    return model


#%% aidl_train
def aidl_train(model, trX, prX, ver_path, config):
    
    File_Log = os.path.join(ver_path, "log.csv")
    File_Model_Save_Last = os.path.join(ver_path, "last_model.h5")
    File_Model_Save_Best = os.path.join(ver_path, "best_model.h5")
    File_Model_Summ = os.path.join(ver_path, "model_summ.txt")
    File_Model_Png = os.path.join(ver_path, "model_png.png")
    File_prX_predict = os.path.join(ver_path, 'prX_pred', "prX_predict.csv")
    engine_path = os.path.join(ver_path, 'engine')

    _EPOCHS = config["_EPOCHS"]
    _PATIENCE = config["_PATIENCE"]
    _BAT_SIZE = config["_BAT_SIZE"]
    
    with open(File_Model_Summ, 'w') as fp:
        model.summary(print_fn=lambda x: fp.write(x + '\n'))

    ### draw plot..
    tf.keras.utils.plot_model(model, to_file=File_Model_Png, show_shapes=True, expand_nested=True, dpi=256)

    ### early stop..
    e_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                min_delta=0.0, patience=_PATIENCE, verbose=0, mode='auto')

    ### callback..
    mcb = myCallBack(model, trX, prX, File_prX_predict)

    ### csvLogger..
    log = tf.keras.callbacks.CSVLogger(File_Log, separator=',', append=False)

    ### best model save..
    sv_best_model = tf.keras.callbacks.ModelCheckpoint(File_Model_Save_Best, monitor='val_loss', mode='min', save_best_only=True)

    ### model checkpoint..
    save_path = os.path.join(engine_path, 'model_{epoch:04d}.h5')
    os.makedirs(ver_path, exist_ok=True)
    sv_model = tf.keras.callbacks.ModelCheckpoint(save_path, save_freq='epoch', period=1)
    
    m_hist = model.fit([trX.Xn], trX.y,
            validation_data=([prX.Xn], prX.y),
            epochs=_EPOCHS,
            batch_size=_BAT_SIZE,
            verbose=1,
            shuffle=True,
            sample_weight=trX.sample_weight, 
            callbacks=[e_stop, mcb, log, sv_best_model, sv_model])

    ### Save model
    model.save(File_Model_Save_Last)
    return model, m_hist


#%% doit
def doit(df_pos_neg_path, config):
    
    global win_size
    win_size = int(df_pos_neg_path.split('/')[-1].split('mer_')[0])
        
    kFold_num = config["kFold_num"]
    kFold_idx = config["kFold_idx"]
    
    trX_path = os.path.join(dataset_path, 'train', f'trX_{win_size}mer_{kFold_num}_{kFold_idx}.pkl')
    prX_path = os.path.join(dataset_path, 'train', f'prX_{win_size}mer_{kFold_num}_{kFold_idx}.pkl')
    if all([os.path.exists(trX_path), os.path.exists(prX_path)]):
        print('already trX, prX file exist!')
        with open(trX_path, 'rb') as f:
            trX = pickle.load(f)
        with open(prX_path, 'rb') as f:
            prX = pickle.load(f)
    else:
        print('make trX, prX file')
        df_trX, df_prX = get_df_trprX(df_pos_neg_path, kFold_num, kFold_idx, use_save=True)
        trX, prX = get_trprX(dataset_path, kFold_num, kFold_idx, df_trX, df_prX, use_save=True)
        del df_trX, df_prX

    print('make trX, prX file done!')
    ver = args.dir_name
    
    ## Run_Tag
    ver_path = os.path.join(script_path, ver)
    if os.path.exists(ver_path):
        print(f"There is already [{ver_path}] dir.")

        print(f'[{ver_path}] is removed.')
        sh.rmtree(ver_path)

    engine_path = os.path.join(ver_path, 'engine')
    os.makedirs(engine_path, exist_ok=True)
        
    ## main code start
    mcc = MCC()
    model = init_model(trX, config)
    model.compile(metrics=[mcc, 
        tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), 
        'acc', tf.keras.metrics.AUC(name='auc'),], 
        loss='binary_crossentropy', optimizer='adam')

    print(model.summary())

    pos_num = 0; neg_num = 0
    for i in trX.y:
        if i == 1: pos_num +=1
        else: neg_num +=1
    print(f'pos: {pos_num}, neg: {neg_num}')

    print("########### config ###########")
    print(config)
    print("##############################")

    model, hist = aidl_train(model, trX, prX, ver_path, config)

    ### save hist and plot.....
    File_Hist_Save = os.path.join(ver_path, "hist.pickle")
    with open(File_Hist_Save, 'wb') as fp:
        pickle.dump(hist.history, fp)

    print("\n\n############### model-trained..##############")


def get_args():
    """
    return arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--df_train_path',  default='', type=str)
    parser.add_argument('-c', '--config_path', default='', type=str)
    parser.add_argument('-d', '--dir_name', default='', type=str)
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu name')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get args
    args = get_args()
    print(f'## ARGS ##\n{args}\n')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    config = getJson(args.config_path)
    df_pos_neg_path = args.df_train_path
    doit(df_pos_neg_path, config)

