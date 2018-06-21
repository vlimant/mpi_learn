import os
import glob

train = glob.glob('/bigdata/shared/LCDJets_Abstract_IsoLep_lt_20/train/*.h5')
test  = glob.glob('/bigdata/shared/LCDJets_Abstract_IsoLep_lt_20/val/*.h5')

open('train_topclass.list','w').write( '\n'.join(sorted( train[:10] )))
open('test_topclass.list','w').write( '\n'.join(sorted( test[:2] )))
