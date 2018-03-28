from mpi_learn.train.GanModel import GANModel
import h5py
import setGPU
import json
import time

gm = GANModel()
gm.compile()

try:
    gm.generator.load_weights('simple_generator.h5')
    gm.discriminator.load_weights('simple_discriminator.h5')
except:
    print ("fresh weights")

files = filter(None,open('train_3d.list').read().split('\n'))
history = {}
etimes=[]
ftimes={}
start = time.mktime(time.gmtime())

for e in range(3): ## epochs
    history[e] = []
    ftimes[e] = []
    e_start = time.mktime(time.gmtime())    
    for f in files:
        f_start = time.mktime(time.gmtime())            
        print ("new file",f,"epoch",e)
        h=h5py.File(f)
        X = h['X']
        cat = h['y']['a']
        E = h['y']['b']
        cellE = h['y']['c']
        
        
        N = X.shape[0]
        bs =100
        start=0
        end = start+bs
        while end<N:
            sub_X = X[start:end]
            sub_Y = [cat[start:end], E[start:end],cellE[start:end]]
            
            losses = gm.train_on_batch(sub_X,sub_Y)
            print (losses)
            #open('simple_train.json','w').write(json.dumps( {'h':history,'et':etimes,'ft':ftimes} ))            
            history[e].append( [list(l) for l in losses] )
            start += bs
            end += bs
        gm.generator.save_weights('simple_generator.h5')
        gm.discriminator.save_weights('simple_discriminator.h5')
        gm.combined.save_weights('simple_combined.h5')
        h.close()
        f_stop = time.mktime(time.gmtime())
        print (f_stop - f_start,"[s] for file",f)
        ftimes[e].append( f_stop - f_start )
        open('simple_train.json','w').write(json.dumps( {'h':history,'et':etimes,'ft':ftimes} ))
        
    e_stop = time.mktime(time.gmtime())
    print (e_stop - e_start,"[s] for epoch",e)
    etimes.append( e_stop - e_start)
