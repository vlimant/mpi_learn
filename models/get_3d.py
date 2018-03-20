import os
import glob
import  h5py
import numpy as np

def get_data(datafile):
    #get data for training
    #print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    y=f.get('target')
    X=np.array(f.get('ECAL'))
    y=(np.array(y[:,1]))
    X[X < 1e-6] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    ecal = np.squeeze(np.sum(X, axis=(1, 2, 3)))
    print X.shape
    print y.shape
    print ecal.shape

    f.close()
    return X, y, ecal


for F in glob.glob('/bigdata/shared/LCD/NewV1/*scan/*.h5'):
    _,d,f = F.rsplit('/',2)
    print d,f
    if not 'Ele' in d: continue
    X = None
    sub_split = 1
    if sub_split==1:
        nf = '/data/shared/3DGAN/%s_%s.h5'%( d,f)
        if os.path.isfile( nf) :
            continue
        if X is None:
            X,y,ecal = get_data(F)
        o = h5py.File(nf,'w')
        o['X'] = X
        o.create_group("y")
        o['y']['a'] = np.ones(y.shape)
        o['y']['b'] = y
        o['y']['c'] = ecal
    else:
        for sub in range(sub_split):
            nf = '/data/shared/3DGAN/%s_%s_sub%s.h5'%( d,f,sub)
            if os.path.isfile( nf) :
                continue
            if X is None:
                X,y,ecal = get_data(F)
                N = X.shape[0]
                splits = [i*N/sub_split for i in range(sub_split)]+[-1]
            o = h5py.File(nf,'w')
            o['X'] = X[splits[sub]:splits[sub+1],...]
            o.create_group("y")
            o['y']['a'] = np.ones(y[splits[sub]:splits[sub+1],...].shape)
            o['y']['b'] = y[splits[sub]:splits[sub+1],...]
            o['y']['c'] = ecal[splits[sub]:splits[sub+1],...]
            o.close()
            X = None
    o.close()

open('train_3d.list','w').write( '\n'.join(filter(lambda f:not 'sub' in f,glob.glob('/data/shared/3DGAN/*.h5')[:-4])))
open('test_3d.list','w').write( '\n'.join(filter(lambda f:not 'sub' in f,glob.glob('/data/shared/3DGAN/*.h5')[-4:])))

open('train_small_3d.list','w').write( '\n'.join(filter(lambda f:not 'sub' in f,glob.glob('/data/shared/3DGAN/*.h5')[:-4])))
open('test_small_3d.list','w').write( '\n'.join(filter(lambda f:not 'sub' in f,glob.glob('/data/shared/3DGAN/*.h5')[-4:])))

open('train_7_3d.list','w').write( '\n'.join(filter(lambda f:not 'sub' in f,glob.glob('/data/shared/3DGAN/*.h5')[:7])))
open('test_1_3d.list','w').write( '\n'.join(filter(lambda f:not 'sub' in f,glob.glob('/data/shared/3DGAN/*.h5')[-1:])))
