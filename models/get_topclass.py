import os
import glob
import sys

dest='/bigdata/shared/LCDJets_Abstract_IsoLep_lt_20'
import socket
host = os.environ.get('HOST', os.environ.get('HOSTNAME',socket.gethostname()))
if 'titan' in host:
    dest='/ccs/proj/csc291/DATA/LCDJets_Abstract_IsoLep_lt_20'
train = glob.glob(dest+'/train/*.h5')
test  = glob.glob(dest+'/val/*.h5')

N=10
Nt=N/5
if len(sys.argv)>=1:
    a = sys.argv[1]
    if a.isdigit():
        N = int(a)
        Nt=N/5            
    else:
        N,Nt = map(int, a.split(','))


open('train_topclass.list','w').write( '\n'.join(sorted( train[:N] )))
open('test_topclass.list','w').write( '\n'.join(sorted( test[:Nt] )))
