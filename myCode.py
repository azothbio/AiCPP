#!/share/anaconda3/envs/py38/bin/python
import tensorflow as tf
import json

class MCC(tf.keras.metrics.Metric):
    def __init__(self, name="mcc", threshold=0.5, **kwargs):
        super(MCC,self).__init__(name=name,**kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.mcc = self.add_weight(name="mcc", initializer="zeros")
        self.threshold = threshold
    
    def update_state(self,y_true,y_pred,sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.where(y_pred>self.threshold,1,0), tf.bool)
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True)), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True)), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False)), tf.float32))
        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False)), tf.float32))
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)
        self.tn.assign_add(tn)
    
    def result(self):
        numer = self.tp*self.tn - self.fp*self.fn
        denumer = tf.math.sqrt((self.tp+self.fp)*(self.tp+self.fn)*(self.tn+self.fp)*(self.tn+self.fn))
        return numer/denumer
    
    def print_confusion(self):
        tp = self.tp.numpy()
        fp = self.fp.numpy()
        fn = self.fn.numpy()
        tn = self.tn.numpy()
        
        print(f"\n[ {tp:^9,.0f} {fp:^9,.0f} ]\n[ {fn:^9,.0f} {tn:^9,.0f} ]\n")


def getJson(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        contents = f.read()
        while "/*" in contents:
            preComment, postComment = contents.split('/*', 1)
            contents = preComment + postComment.split('*/', 1)[1]
        return json.loads(contents.replace("'", '"'))
