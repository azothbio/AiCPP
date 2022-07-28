#!/share/anaconda3/envs/pyk/bin/python
"""
aiCPP2_r4.py:
    Not using Attention..
    Different version based on r1.

aiCPP2_r3.py:
    Different version based on r1.

aiCPP2_r2.py:
    Different version based on r1.

aiCPP2_r1.py:
    r1

    jms@AZB, 2021-07-13
"""
import os, sys
import numpy as np
import pickle
from get_trprX import MXY, KXY, get_trprX

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kL
from tensorflow.keras import models as kM

### sklearn
from sklearn.metrics import r2_score, roc_auc_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef as MCC

gpus = tf.config.experimental.list_physical_devices('GPU')
print("##gpus=", gpus)
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("##TF>set gpu-memory-growth 'True'...")
    except RuntimeError as e:
        print(e)


## learning prms
_EPOCHS = 9000
_BAT_SIZE = 1024*100
#_FP_LEN = 16
_DIM_EMB = 3
_DIM_LAT = 3
_DIM_CNN = 3
_N_CNN_Xall = 0
_N_KERNEL = 4
_KERNEL_SIZE = 9
_STRIDES = 1
_DROP_OUT = 0.3

_PATIENCE = 100


## Run tag
np.random.seed(101)
Run_Tag = 'r4'
if not os.path.exists(Run_Tag):
    os.makedirs(Run_Tag)
File_Model_Save = "%s/model_save.h5" % Run_Tag
File_Model_Sum = "%s/model_sum.txt" % Run_Tag
File_Model_Png = "%s/model_shape.png" % Run_Tag
File_Prd = "%s/prd.csv" % Run_Tag
File_Hist = "%s/hist.pickle" % Run_Tag
File_Log = "%s/log.csv" % Run_Tag
File_Log_png = "%s/log.png" % Run_Tag


## Callback
class myCallBack(keras.callbacks.Callback):
    def __init__(self, model, prX): 
        self.model = model 
        self.prX = prX 
        with open(File_Log, 'w') as fp:
            fp.write("#epoch,loss,acc,val_loss,val_acc\n")

    def save_prd(self, epoch, ty, fx, mcc, pauc, logs={}):
        fp = open(File_Prd, 'w')
        fp.write("#Prd>%d,mcc=%.4f,pauc=%.4f,val_loss=%.3f,val_acc=%.3f\n" % (epoch, mcc, pauc, logs['val_loss'], logs['val_acc']))
        for j in range(len(ty)):
            fp.write("%s>%d,%.3f\n" % (self.prX.seq[j], ty[j], fx[j]))
        fp.flush()
        fp.close()

    def append_log(self, epoch, mcc, pauc, logs={}):
        with open(File_Log, 'a') as fp:
            fp.write("%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % \
                    (epoch, mcc, pauc, logs['loss'],logs['acc'],
                        logs['val_loss'], logs['val_acc'])
                    )
    def on_epoch_end(self, epoch, logs={}):
        #print("JMS>>>>logs=", logs)
        self.model.save(File_Model_Save)
        y = self.prX.y
        X = self.prX
        fx = self.model.predict(X.X)
        py = np.where(fx>0.5, 1, 0)
        mcc = MCC(y, py)
        pauc = roc_auc_score(y, fx)
        self.save_prd(epoch, y, fx, mcc, pauc, logs)
        self.append_log(epoch, mcc, pauc, logs)
        cmat = confusion_matrix(y, py)
        print("#\tpauc=%.4f, mcc=%.4f, cmat==>" % (pauc, mcc))
        print(cmat)



## cnn
def myCnn(tag, x, nf=_N_KERNEL, ks=_KERNEL_SIZE, st=_STRIDES, ps=2, act='relu', pad='valid'):
    x = kL.Conv1D(filters=nf, kernel_size=ks, strides=st, padding=pad, activation=act, name='%s_cnn' % tag)(x)
    if ps > 1:
        x = kL.MaxPool1D(pool_size=ps, name="%s_pool" % tag)(x)
    x = kL.Dropout(_DROP_OUT, name="%s_drop" % tag)(x)
    return x

## Attention
def attN(tag, xQ, xV):
    xA = kL.Attention(name="%s_attn" % tag, use_scale=True, dropout=_DROP_OUT)([xQ, xV])
    #xA = kL.Permute((2,1))(xA)
    #xC = kL.Concatenate(axis=-2, name="%s_concat" % tag)([xQ, xA])
    #print("###JMS>xQ.shape=", xQ.shape)
    #print("###JMS>xV.shape=", xV.shape)
    #print("###JMS>xA.shape=", xA.shape)
    xC = kL.Add(name='%s_add' % tag)([xQ, xA])
    return xA, xC

def plot_log(history):
    import matplotlib.pyplot as plt
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(history['loss'], 'b', label='tr_loss')
    loss_ax.plot(history['val_loss'], 'r', label='pr_loss')
    loss_ax.set_xlabel('epochs')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='lower center')

    acc_ax.plot(history['acc'], 'b', label='tr_acc')
    acc_ax.plot(history['val_acc'], 'r', label='tr_acc')
    acc_ax.set_ylabel('acc')
    acc_ax.legend(loc='upper center')

    plt.grid()
    plt.savefig(File_Log_png, dpi=300)
    plt.show()


def aidl_train(model, trX, prX):
    ## Save model-shape
    with open(File_Model_Sum, 'w') as fp:
        model.summary(print_fn=lambda x: fp.write(x + '\n'))
    keras.utils.plot_model(model, to_file=File_Model_Png, show_shapes=True, expand_nested=True, dpi=256)

    ## callback
    mcb = myCallBack(model, prX)
    e_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
            min_delta=0.001, patience=_PATIENCE,
            verbose=0,
            mode='auto'
            )

    hist = model.fit(trX.X, trX.y, 
            validation_data=(prX.X, prX.y),
            #validation_split = 0.2,
            validation_freq = 1,
            epochs=_EPOCHS,
            batch_size=_BAT_SIZE,
            verbose=1,
            shuffle=True,
            class_weight = {0:1, 1:900},
            callbacks=[e_stop, mcb],
            )

    model.save(File_Model_Save)
    return model, hist


def init_model(trX):
    Xinp_n = kL.Input(shape=trX.X.shape[1:], name='Xinp_n')
    
    l_emb = kL.Embedding(input_dim=21, output_dim=_DIM_EMB, name='l_emb')
    xemb = l_emb(Xinp_n)
    print("##JMS-Chk1>xemb.shape=", xemb.shape)

    ## c2
    xc2 = myCnn('cnn-2', xemb, nf=_DIM_CNN, ks=2, st=1, ps=1, act='relu')
    xc3 = myCnn('cnn-3', xemb, nf=_DIM_CNN, ks=3, st=1, ps=1, act='relu')
    xc4 = myCnn('cnn-4', xemb, nf=_DIM_CNN, ks=4, st=1, ps=1, act='relu')
    xc5 = myCnn('cnn-5', xemb, nf=_DIM_CNN, ks=5, st=1, ps=1, act='relu')

    xA1, xC1 = attN('att1', xemb, xemb)
    xA2, xC2 = attN('att2', xemb, xc2)
    xA3, xC3 = attN('att3', xemb, xc3)
    xA4, xC4 = attN('att4', xemb, xc4)
    xA5, xC5 = attN('att5', xemb, xc5)

    xA = kL.Concatenate(axis = -2, name='concat_xA')([xA1, xA2, xA3, xA4, xA5])
    xC = kL.Concatenate(axis = -2, name='concat_xC')([xC1, xC2, xC3, xC4, xC5])
    print("##JMS-Chk2>xA.shape=", xA.shape)
    print("##JMS-Chk3>xC.shape=", xC.shape)

    xcA = myCnn('cnn-A', xA, nf=_DIM_CNN, ks=2, st=2, ps=1, act='relu')
    xcC = myCnn('cnn-C', xA, nf=_DIM_CNN, ks=2, st=2, ps=1, act='relu')

    x = kL.Concatenate(axis = -2, name='concat_xAC')([xcA, xcC])
    """
    xA, xC = attN('att1', x, xemb)
    print("##JMS-Chk3>xA.shape=", xA.shape)
    print("##JMS-Chk4>xC.shape=", xC.shape)



    xA, xC = attN('att2', xemb, xemb)
    print("##JMS-Chk3>xA.shape=", xA.shape)
    print("##JMS-Chk4>xC.shape=", xC.shape)

    xemb = kL.Flatten(name='flat_xemb')(xemb)
    xA = kL.Flatten(name='flat_xA')(xA)
    xC = kL.Flatten(name='flat_xC')(xC)
    print("##JMS-Chk5.1>xemb.shape=", xemb.shape)
    print("##JMS-Chk5>xA.shape=", xA.shape)
    print("##JMS-Chk6>xC.shape=", xC.shape)

    xemb = kL.Dense(_DIM_LAT*1, activation='relu', name='d1_xemb')(xemb)
    xA = kL.Dense(_DIM_LAT*1, activation='relu', name='d1_xA')(xA)
    xC = kL.Dense(_DIM_LAT*1, activation='relu', name='d1_xC')(xC)
    print("##JMS-Chk7.1>xemb.shape=", xemb.shape)
    print("##JMS-Chk7>xA.shape=", xA.shape)
    print("##JMS-Chk8>xC.shape=", xC.shape)

    x = kL.Concatenate(name='concat_xAC')([xemb, xA, xC])
    """
    x = myCnn('cnn-x', x, nf=_DIM_CNN*2, ks=5, st=2, ps=2, act='relu')
    x = kL.Flatten(name='flat_x')(x)
    x = kL.Dropout(_DROP_OUT)(x)
    print("##JMS-Chk9>x.shape=", x.shape)

    x = kL.Dense(_DIM_LAT*2, activation='relu', name='d1_x')(x)
    x = kL.Dropout(_DROP_OUT)(x)
    print("##JMS-Chk10>x.shape=", x.shape)

    x = kL.Dense(_DIM_LAT, activation='relu', name='d2_x')(x)
    x = kL.Dropout(_DROP_OUT)(x)
    print("##JMS-Chk11>x.shape=", x.shape)

    fx = kL.Dense(1, activation='sigmoid', name='fx')(x)

    model = kM.Model(inputs=Xinp_n, outputs=fx, name='model_cpp2')
    model.compile(
            loss = 'binary_crossentropy',
            optimizer='adam',
            metrics=['acc', keras.metrics.AUC()],
            )

    return model



def doit():
    trX, prX = get_trprX(200, 1)
    trX.Info()
    prX.Info()


    model = init_model(trX)
    print(model.summary())


    model, hist = aidl_train(model, trX, prX)

    
    with open(File_Hist, 'wb') as fp:
        pickle.dump(hist.history, fp)
    plot_log(hist.history)


    ### test reloaded model..
    m = kM.load_model(File_Model_Save)
    print("#"*20 + " Model-reloaded " + "#"*20)
    print(m.summary())
    print("#"*20 + " Model-reloaded " + "#"*20)



if __name__ == '__main__':
    doit()



