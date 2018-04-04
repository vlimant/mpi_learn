from mpi_learn.train.GanModel import GANModel
import h5py
import setGPU
import json
import time

gm = GANModel()

reload = False
fresh = True
tag=''
if reload:
    tag+='reload_'
    print ("Reloading")

    p = False
    lr = 0.01
    if not p:
        gm.compile( prop = False, lr=lr)
        tag+='sgd%s_'%lr
    else:
        gm.compile( prop = True)
        tag+='rmsprop_'

    ## start from an exiting model
    gm.generator.load_weights('FullRunApr3/simple_generator.h5')
    gm.discriminator.load_weights('FullRunApr3/simple_discriminator.h5')
    gm.combined.load_weights('FullRunApr3/simple_combined.h5')
else:
    lr = 0.0
    if lr:
        gm.compile(prop = False, lr=lr)
    else:
        gm.compile()

    if not fresh:
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


train_me = True
over_test= True
max_batch = 10
ibatch=0
for e in range(3): ## epochs
    history[e] = []
    ftimes[e] = []
    e_start = time.mktime(time.gmtime())
    if max_batch and ibatch>max_batch:
        break
    for f in files:
        f_start = time.mktime(time.gmtime())            
        print ("new file",f,"epoch",e)
        h=h5py.File(f)
        X = h['X']
        cat = h['y']['a']
        E = h['y']['b']
        cellE = h['y']['c']
        
        
        N = X.shape[0]
        bs =200
        start=0
        end = start+bs
        while end<N:
            if max_batch and ibatch>max_batch:
                break
            sub_X = X[start:end]
            sub_Y = [cat[start:end], E[start:end],cellE[start:end]]
            ibatch+=1
            print (ibatch,ibatch>max_batch,max_batch)
            if over_test or not train_me:
                t_losses = gm.test_on_batch(sub_X,sub_Y)
                print (t_losses)            
            if train_me:
                losses = gm.train_on_batch(sub_X,sub_Y)

            #print (losses)
            history[e].append( [list(l) for l in losses] )
            start += bs
            end += bs
        gm.generator.save_weights('simple_generator_%s.h5'%tag)
        gm.discriminator.save_weights('simple_discriminator_%s.h5'%tag)
        gm.combined.save_weights('simple_combined_%s.h5'%tag)
        h.close()
        f_stop = time.mktime(time.gmtime())
        print (f_stop - f_start,"[s] for file",f)
        ftimes[e].append( f_stop - f_start )
        open('simple_train_%s.json'%tag,'w').write(json.dumps( {'h':history,'et':etimes,'ft':ftimes} ))
        if max_batch and ibatch>max_batch:
            break

        
    e_stop = time.mktime(time.gmtime())
    print (e_stop - e_start,"[s] for epoch",e)
    etimes.append( e_stop - e_start)
