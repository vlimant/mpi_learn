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
    X,y,ecal = get_data(F)
    o = h5py.File('/data/shared/3DGAN/%s_%s.h5'%( d,f), 'w')
    #o.create_group("X")
    #o['X']['a'] = X
    #o['X']['b'] = ecal
    o['X'] = X
    o.create_group("y")
    #o['y'] = y
    o['y']['a'] = np.ones(y.shape)
    o['y']['b'] = y
    o['y']['c'] = ecal
    
    o.close()
