#!/share/anaconda3/envs/pyk/bin/python
import numpy as np
import pickle
from dlxy.dlxy import DLXY

from sklearn.model_selection import KFold



Fn_Input = "../datasets/cpp_9mer_all.csv"

def dump_dlx():
    dlx = DLXY('cpp2_9mers', dtype=np.int8)
    fp = open(Fn_Input, 'r')
    while 1:
        line = fp.readline()
        if not line:
            break
        if line[0] == '#':
            continue
        ll = line.split(',')
        dlx.ChkIn(ll[0].strip(), int(ll[1].strip()))

    fp.close()
    dlx.Pack(selection='first')

    dlx.Info()

    dlx.Dump('cpp2_9mers')


def reload_dlx():
    dlx = DLXY("cpp2_9mers", dtype=np.int8)
    dlx.Reload("cpp2_9mers")

    """
#
#DLXY>cpp2_9mers, bool_pack=1, ##note=reloaded, dtype= <class 'numpy.int8'>
#	#ChkIn>l_lbl=11371445, d_x=11367198
#	#Pack>lc_lbl=11367198, dc_l2i=11367198
#	#Pack>X.shape= (11367198,)
dlx.X.sum= 7853
    """
    return dlx

#### Sedtup d_a2i
AA_CHARS =  "ACDEFGHIKLMNPQRSTVWY-"  
_d_a2i = {}
for i,a in enumerate(AA_CHARS):
    _d_a2i[a] = i


class MXY(object):
    def __init__(self, tag, dlx):
        self.tag = tag
        self.y = dlx.X
        self.seq = []
        ## set x
        lX = []
        for i,seq in enumerate(dlx.lc_lbl):
            lx = [_d_a2i[a] for a in seq]
            lX.append(lx)
            self.seq.append(seq)
        self.X = np.array(lX, dtype=np.int8)

    def Info(self, note='note'):
        print("##MXY>%s, note=%s, n=%d, sum_Y=%.1f" % (self.tag, note, len(self.y), self.y.sum()))
        print("\tX.shape=", self.X.shape)


class KXY(object):
    def __init__(self, tag, idx, mxy):
        self.tag = tag
        self.y = mxy.y[idx]
        self.X = mxy.X[idx]
        self.seq = [mxy.seq[j] for j in idx]

    def Info(self, note='note'):
        print("##KXY>%s, note=%s, n=%d, sum_Y=%.1f" % (self.tag, note, len(self.y), self.y.sum()))
        print("\tX.shape=", self.X.shape)



def get_trprX(nk=10, idx_k=0):
    #dump_dlx()

    """
    ##### reload dlx
    dlx = reload_dlx()

    ##### build mxy
    mxy = MXY('master', dlx)

    
    ##### dump mxy
    fp = open("./mxy/mxy.pickle", 'wb')
    pickle.dump(mxy, fp)
    fp.close()
    """

    ##### reload mxy
    fp = open("./mxy/mxy.pickle", 'rb')
    mxy = pickle.load(fp)
    fp.close()

    
    ##### prep trprX

    idx_arr = np.arange(len(mxy.y))
    kf = KFold(n_splits=nk, shuffle=True)
    lkf = list(kf.split(idx_arr))
    ktr, kpr = lkf[idx_k]

    trX = KXY('trX', ktr, mxy)
    prX = KXY('prX', kpr, mxy)

    return trX, prX


def doit():
    trX, prX = get_trprX(50, 1)
    trX.Info()
    prX.Info()


if __name__ == '__main__':
    doit()
