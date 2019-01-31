import os
import glob
dest='/bigdata/shared/LCDJets_Abstract_IsoLep_lt_20'
import socket
host = os.environ.get('HOST', os.environ.get('HOSTNAME',socket.gethostname()))
if 'titan' in host:
    dest='/ccs/proj/csc291/DATA/LCDJets_Abstract_IsoLep_lt_20'
train = glob.glob(dest+'/train/*.h5')
test  = glob.glob(dest+'/val/*.h5')

open('train_topclass.list','w').write( '\n'.join(sorted( train[:10] )))
open('test_topclass.list','w').write( '\n'.join(sorted( test[:2] )))
