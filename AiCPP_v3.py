#%% import module

from u_emb import EXW, get_l_fas_from_df
from tqdm import tqdm
from myCode import MCC, getJson

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kL
from tensorflow.keras import models as kM
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Reshape, Dropout
from tensorflow.keras.layers import Bidirectional

## sklearn metrics
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, r2_score

import os, sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

import re
import regex
import math
from Bio.SeqIO.FastaIO import SimpleFastaParser

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
    
    df_trX = df_trX[~(df_trX['seq'].str.contains('.*[BJOUXZ].*'))]
    df_prX = df_prX[~(df_prX['seq'].str.contains('.*[BJOUXZ].*'))]

    df_trX.loc[:, 'Xn'] = df_trX['seq'].apply(lambda x: get_Xn(d_aa2i, x))
    df_trX.loc[:, 'Xc'] = df_trX['Xn'].apply(lambda x: get_Xc(x))
    
    df_prX.loc[:, 'Xn'] = df_prX['seq'].apply(lambda x: get_Xn(d_aa2i, x))
    df_prX.loc[:, 'Xc'] = df_prX['Xn'].apply(lambda x: get_Xc(x))

    return df_trX, df_prX

class XY_data(object):
    def __init__(self, df, aachars="ACDEFGHIKLMNPQRSTVWY-"):
        self.seq = list(df['seq'])
        self.Xn = np.array(list(df['Xn']))
        self.Xc = np.array(list(df['Xc']))
        self.y = np.array(list(df['y']))

#%% get df_trX, df_prX
def get_df_trprX(dataset_path, kFold_num, kFold_idx, use_save=False):
    
    def df_row_copy(df, copy_num):
        l_seq = list(df['seq']) * copy_num
        l_y = list(df['y']) * copy_num
        df = pd.DataFrame([i for i in zip(l_seq, l_y)], columns=['seq', 'y'])
        return df

    df_pos_path = os.path.join(dataset_path, '9mer_clean', 'cpp_9mer_pos_7853.csv')
    df_neg_path = os.path.join(dataset_path, '9mer_clean', 'cpp_9mer_neg_15169.csv')
    df_decoy_path = os.path.join(dataset_path, '9mer_clean', 'cpp_9mer_decoy_11344176.csv')
    assert os.path.exists(df_pos_path) and os.path.exists(df_neg_path) and os.path.exists(df_decoy_path), f"There is no path! Check next 3 paths\n{df_pos_path}\n{df_neg_path}\n{df_decoy_path}"
    
    df_all_path = os.path.join(dataset_path, '9mer_clean', 'cpp_9mer_all.csv')
    df_pos_neg_path = os.path.join(dataset_path, '9mer_clean', 'cpp_9mer_pos_neg.csv')
    df_trX_path = os.path.join(dataset_path, 'train', f'cpp_trX_{kFold_num}_{kFold_idx}.csv')
    df_prX_path = os.path.join(dataset_path, 'train', f'cpp_prX_{kFold_num}_{kFold_idx}.csv')
    df_prX_without_decoy_path = os.path.join(dataset_path, 'train', f'cpp_prX_{kFold_num}_{kFold_idx}_without_decoy.csv')
    
    # check [ df_pos_neg.csv ] file
    if not os.path.exists(df_all_path) or os.path.exists(df_pos_neg_path):
        df_pos = pd.read_csv(df_pos_path, index_col=False)
        df_neg = pd.read_csv(df_neg_path, index_col=False)

        df_pos_neg = pd.concat([df_pos, df_neg])
        df_pos_neg.to_csv(df_pos_neg_path, index=False)

    # check [ df_trX.csv, df_prX.csv ] file
    if all([os.path.exists(df_trX_path), os.path.exists(df_prX_path)]):
        df_trX = pd.read_csv(df_trX_path, index_col=False)
        df_prX = pd.read_csv(df_prX_path, index_col=False)

    else:
        # split df_trX, df_prX by using KFold
        kf = StratifiedKFold(kFold_num, shuffle=True, random_state=101)
        df_pos_neg = pd.read_csv(df_pos_neg_path)
        result = [i for i in kf.split(df_pos_neg, df_pos_neg['y'])]
        df_trX = df_pos_neg.iloc[result[kFold_idx][0]]
        df_prX = df_pos_neg.iloc[result[kFold_idx][1]]
        df_prX = df_prX.drop_duplicates('seq')
        if use_save:
            df_prX.to_csv(df_prX_without_decoy_path)
        
        
        # add decoy
        df_decoy = pd.read_csv(df_decoy_path, index_col=False)
        df_decoy = df_decoy[~(df_decoy['seq'].isin(set(df_pos_neg[df_pos_neg['y']==1]['seq'])))]

        # neg_num = df_prX['y'].value_counts()[0]

        df_decoy = df_decoy.sample(frac=0.5)
        df_decoy_prX = df_decoy.sample(frac=0.02)
        df_decoy_trX = df_decoy[~(df_decoy['seq'].isin(df_decoy_prX['seq']))]

        df_trX_neg = df_row_copy(df_trX[df_trX['y']==0], 49) # MLCPP negative set * 100
        df_trX = pd.concat([df_trX, df_trX_neg, df_decoy_trX])
        df_prX = pd.concat([df_prX, df_decoy_prX])

        if use_save:
            df_trX.to_csv(df_trX_path)
            df_prX.to_csv(df_prX_path)
            
    return df_trX, df_prX
    
def get_trprX(dataset_path, kFold_num, kFold_idx, df_trX, df_prX, use_save=False):
    trX_path = os.path.join(dataset_path, f'trX_{kFold_num}_{kFold_idx}.pkl')
    prX_path = os.path.join(dataset_path, f'prX_{kFold_num}_{kFold_idx}.pkl')
    
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
    
    #@tf.function
    #def calc_mcc(self, tp, fp, fn, tn):
    #    try: 
    #        return (tp*tn - fp*fn) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    #    except:
    #        return float('Nan')
        
    def save_prd(self, epoch, y, fx):
        save_path = f"self.prX_pred_{epoch}"
        with open(save_path, 'w') as f:
            print("y,fx", file=f)
            for idx, _y in enumerate(y):
                print(f"{self.prX.seq},{_y:0.1f},{fx[idx]:0.3f}", file=f)

    def on_epoch_end(self, epoch, logs={}):
        #self.model.evaluate(self.prX.Xw, self.prX.y)
        #fx = list(self.model.predict([self.prX.Xn, self.prX.Xc]))
        #fx = [y for x in fx for y in x]
        #y = list(self.prX.y)
        self.model.metrics[1].print_confusion()
        #self.save_prd(epoch, y, fx)

#%% Functional-API 

## cnn..
def myCnn(tag, x, nf=8, ks=3, st=2, act='relu', pad='valid'):
    ## x ==> (seq_len, channel)
    x = kL.Conv1D(filters=nf, kernel_size=ks, strides=st, padding=pad, activation=act, name='%s_cnn' % tag)(x)
    return x

def resblock(tag, x, lat_dim):
    fx = myCnn(f"{tag}_1", x, lat_dim, ks=3, st=1, act=None, pad='same')
    fx = kL.BatchNormalization()(fx)
    fx = kL.Activation('elu')(fx)
    
    fx = myCnn(f"{tag}_2", fx, lat_dim, ks=3, st=1, act=None, pad='same')
    fx = kL.BatchNormalization()(fx)
    fx = kL.Add()([x,fx])
    out = kL.Activation('elu')(fx)
    
    return out

def identity_block(tag, x, nf=8, ks=3):
    x_shortcut=x
    x = kL.Conv1D(filters=nf, kernel_size=1, strides=1, padding='valid', name=f'{tag}_c1d_1a')(x)
    x = kL.BatchNormalization(name=f'{tag}_bn_1b')(x)
    x = kL.Activation('relu')(x)
    x = kL.Conv1D(filters=nf, kernel_size=ks, strides=1, padding='same', name=f'{tag}_c1d_2a')(x)
    x = kL.BatchNormalization(name=f'{tag}_bn_2b')(x)
    x = kL.Activation('relu')(x)
    x = kL.Conv1D(filters=nf, kernel_size=1, strides=1, padding='valid', name=f'{tag}_c1d_3a')(x)
    x = kL.BatchNormalization(name=f'{tag}_bn_3b')(x)
    x = kL.Add()([x,x_shortcut])
    x = kL.Activation('relu')(x)
    return x

## Attention ==> (xV, xQ) ==>
## return xA, xC
def attN(tag, xV, xQ, dropout):
    ## xV = (seq_dim, lat_dim)
    ## xQ = (seq_dim, lat_dim)
    xA = kL.Attention(name='%s_att' % tag, use_scale=True, dropout=dropout)([xV, xQ])
    xC = kL.Add(name='%s_con' % tag)([xQ, xA])
    return xA, xC
    #return xC


def att_tst(tag, xV, xQ, dropout):
    """
    return Attn..
    """
    xA = kL.Attention(name='%s_att_tst' % tag, use_scale=True, dropout=dropout)([xV, xQ])
    return xA

def myDense(inputX, nodeNum, activation='relu', dropout=0.1, use_name=False):
    if use_name:
        x = kL.Dense(nodeNum, name=use_name)(inputX)
    else:
        x = kL.Dense(nodeNum)(inputX)
    x = kL.BatchNormalization()(x)
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

    _TOK_DIM = config["_TOK_DIM"]
    _EMB_DIM = config["_EMB_DIM"]
    _LAT_DIM = config["_LAT_DIM"]
    # _DROP_OUT = config["_DROP_OUT"]
    
    dim_xn = trX.Xn.shape[1:]
    dim_xc = trX.Xc.shape[1:]
    
    Xinp_xn = kL.Input(shape=dim_xn, name='Xinp_xn')
    Xinp_xc = kL.Input(shape=dim_xc, name='Xinp_xc')

    emb_xn1 = kL.Embedding(21, 8, name='emb_xn1')(Xinp_xn)
    emb_xn2 = kL.Embedding(21, 8, name='emb_xn2')(Xinp_xn)
    print(f"emb_xn1.shape : {emb_xn1.shape}")
    print(f"emb_xn2.shape : {emb_xn2.shape}")
    
    #### LSTM
    lstm_x1 = kL.Bidirectional(kL.LSTM(4, return_sequences=True, dropout=0.2, name='lstm_xn'))(emb_xn1)
    lstm_x2 = kL.Bidirectional(kL.LSTM(4, return_sequences=True, dropout=0.2, name='lstm_xc'))(emb_xn2)

    #### residual CNN
    #xn11 = resblock('xn11', emb_xn1, 10)
    #xn12 = resblock('xn12', xn11, 10)
    #xn13 = resblock('xn13', xn12, 10)
    #print(f"xn11.shape : {xn11.shape}")
    #print(f"xn12.shape : {xn12.shape}")
    #print(f"xn13.shape : {xn13.shape}")

    #xn21 = identity_block('xn21', emb_xn2, 10)
    #xn22 = identity_block('xn22', xn21, 10)
    #xn23 = identity_block('xn23', xn22, 10)
    #print(f"xn21.shape : {xn21.shape}")
    #print(f"xn22.shape : {xn22.shape}")
    #print(f"xn23.shape : {xn23.shape}")

    #### self_attention
    #x1_A, x1_C = attN('x1', xn11, xn11, dropout=0.1)
    #x1_A, x1_C = attN('x1', lstm_x1, lstm_x1, dropout=0.1)
    #x2_A, x2_C = attN('x2', xn11, lstm_x2, dropout=0.1)
    #x3_A, x3_C = attN('x3', lstm_x2, lstm_x2, dropout=0.1)
    #print(f"x1_A.shape : {x1_A.shape}")
    #print(f"x2_A.shape : {x2_A.shape}")

    x1 = myDense(lstm_x1, 8, activation='elu', dropout=0.2, use_name='dn-1')
    x2 = myDense(lstm_x2, 8, activation='elu', dropout=0.2, use_name='dn-2')
    x = kL.Concatenate(axis=-1, name='concat-1')([x1, x2])

    ### flatten
    x = kL.Flatten()(x)
    x = kL.BatchNormalization()(x)
    #x2 = kL.Flatten()(x2_C)
    #x2 = kL.BatchNormalization()(x2)
    #x3 = kL.Flatten()(x3_C)
    #x3 = kL.BatchNormalization()(x3)

    ### Concat
    #x = kL.Concatenate(axis=-1, name='concat-1')([x1, x2, x3])
    
    #### LSTM
    #lstm_x = kL.Bidirectional(kL.LSTM(3, return_sequences=True, dropout=0.1, name='lstm_x'))(x)
    
    ### Concat
    #x = kL.Concatenate(axis=-1, name='concat-1')([lstm_xn, lstm_xc])
    
    #x = kL.Concatenate(axis=-1, name='concat-1')([x12, x13, x14, x23, x24, x34, x41, x42, x43])
    #x = kL.Flatten(name='flat')(lstm_x)
    #x = kL.BatchNormalization()(x)
    x = myDense(x, 8, activation='elu', dropout=0.2, use_name='dn-3')
    x = myDense(x, 8, activation='elu', dropout=0.1, use_name='dn-4')
    
    fx = kL.Dense(1, activation='sigmoid', name='fx')(x)

    model = kM.Model(inputs=[Xinp_xn, Xinp_xc], outputs=fx)
        
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
    weight_1 = config["classweight_1"]
    weight_0 = config["classweight_0"]
    
    with open(File_Model_Summ, 'w') as fp:
        model.summary(print_fn=lambda x: fp.write(x + '\n'))
    keras.utils.plot_model(model, to_file=File_Model_Png, show_shapes=True, expand_nested=True, dpi=256)

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
    
    m_hist = model.fit([trX.Xn, trX.Xc], trX.y,
            validation_data=([prX.Xn, prX.Xc], prX.y),
            epochs=_EPOCHS,
            batch_size=_BAT_SIZE,
            verbose=1,
            shuffle=True,
            class_weight={1:weight_1, 0:weight_0},
            callbacks=[e_stop, mcb, log, sv_best_model, sv_model])

    ### Save model
    model.save(File_Model_Save_Last)
    return model, m_hist

#%% doit

def doit(config):
    
    init_path = config["init_path"]
    script_path = os.path.join(init_path, 'scripts')
    dataset_path = os.path.join(init_path, 'dataset')
    assert os.path.exists(script_path) and os.path.exists(dataset_path), f"There is no path! Check next two path\n{script_path}\n{dataset_path}"
    
    kFold_num = config["kFold_num"]
    kFold_idx = config["kFold_idx"]
    df_trX, df_prX = get_df_trprX(dataset_path, kFold_num, kFold_idx, use_save=False)
    trX, prX = get_trprX(dataset_path, kFold_num, kFold_idx, df_trX, df_prX, use_save=False)

    ver = config["version"]
    
    ## Run_Tag
    ver_path = os.path.join(script_path, ver)
    if os.path.exists(ver_path):
        print(f"There is already [{ver_path}] dir. Please check it!!")
        sys.exit()
    else:
        engine_path = os.path.join(ver_path, 'engine')
        os.makedirs(engine_path, exist_ok=True)
        
    File_Hist_Save = os.path.join(ver_path, "hist.pickle")
    File_Model_Save_Last = os.path.join(ver_path, "last_model.h5")
    
    ## main code start
    mcc = MCC()
    #mirrored_strategy = tf.distribute.MirroredStrategy()
    #with mirrored_strategy.scope():
    model = init_model(trX, config)
    model.compile(metrics=[mcc, 
        tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), 
        'acc', tf.keras.metrics.AUC(name='auc'),], 
        loss='binary_crossentropy', optimizer='adam')

    #model = load_model('../engine/20210316_203956_w15/model_2550.h5')
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
    with open(File_Hist_Save, 'wb') as fp:
        pickle.dump(hist.history, fp)

    print("\n\n############### model-trained..##############")


#%%

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

 
#%% __name__ == '__main__'

if __name__ == '__main__':

    config = getJson('config.json')
    doit(config)
