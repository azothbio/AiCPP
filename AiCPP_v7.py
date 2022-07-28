import os
import numpy as np
import tensorflow as tf

from model import init_model

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


BATCH_SIZE = 50000

def get_path(kf):
    VER = "cpp_l32_9"
    Path = "./run/%s" % VER
    if not os.path.exists(Path):
        os.makedirs(Path)

    Path1 = Path+f'/{VER}_{kf}'
    if not os.path.exists(Path1):
        os.makedirs(Path1)
    File_Model = os.path.join(Path1, "model.h5")
    File_Log = os.path.join(Path1, "log.csv")
    File_Prd = os.path.join(Path1, "prd.csv")
    File_Model_Summ = os.path.join(Path1, "model_summ.txt")
    File_Model_Png = os.path.join(Path1, "model_summ.png")
    return Path1, File_Model, File_Log,File_Prd, File_Model_Summ, File_Model_Png

def train():
    tp1= np.load("tp1_21.npy") # feature
    tp2= np.load("tp2_21.npy") # label
    print(np.shape(tp1))
    X,y=(tp1,tp2)
    tp1_ds=tf.data.Dataset.from_tensor_slices((X,y))
    tp1_ds_shu = tp1_ds.shuffle(81920, seed=42)
    train = tp1_ds_shu.enumerate().filter(lambda X,y: X % 20 !=0).map(lambda X,y:y)
    test = tp1_ds_shu.enumerate().filter(lambda X,y: X % 20 ==0).map(lambda X,y:y)
    tr1=train.shuffle(8192).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    pr1=test.shuffle(8192).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    Path, File_Model, File_Log,File_Prd, File_Model_Summ, File_Model_Png = get_path("test")

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        mcb=[
            tf.keras.callbacks.EarlyStopping(patience=1000), 
            tf.keras.callbacks.ModelCheckpoint(filepath=Path, monitor='val_accuracy', mode='max',save_best_only=True), 
            tf.keras.callbacks.CSVLogger(File_Log),
            tf.keras.callbacks.TensorBoard(log_dir=Path),]

        METRICS = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            "accuracy",
            tf.keras.metrics.AUC(name='auc'),]
    
        model = init_model(8,4,16,32)
        model.compile("adam", "binary_crossentropy", metrics=METRICS)

    ##### Save model_summ ###
    with open(File_Model_Summ, 'w') as fp:
        model.summary(print_fn=lambda x: fp.write(x + '\n'))

    m_hist = model.fit(tr1,
            validation_data=pr1,
            epochs=2000, verbose=1,
            steps_per_epoch=None,
            validation_steps=None,
            class_weight={0:1, 1:1000},
            callbacks=[mcb])
    
    model.save(File_Model)


if __name__ == '__main__':
    train()
