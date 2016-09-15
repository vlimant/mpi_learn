import argparse

import Model

parser = argparse.ArgumentParser()
parser.add_argument('model_name', help='model to construct')
args = parser.parse_args()
model_name = args.model_name

model = Model.make_model( model_name )

weights_filename = "%s_weights.h5" % model_name 
model.save_weights( weights_filename, overwrite=True )
print "Saved model weights to %s" % weights_filename

arch_filename = "%s_arch.json" % model_name
model_arch = model.to_json()
with open( arch_filename, 'w' ) as arch_file:
    arch_file.write( model_arch )
print "Saved model architecture to %s" % arch_filename
