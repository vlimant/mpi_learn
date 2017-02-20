### Builds one of the available models.  
# Saves model architecture to <model_name>_arch.json
# and model weights to <model_name>_weights.h5

import argparse

from models.Models import make_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='model to construct')
    args = parser.parse_args()
    model_name = args.model_name

    model = make_model( model_name )

    weights_filename = "%s_weights.h5" % model_name 
    model.save_weights( weights_filename, overwrite=True )
    print ("Saved model weights to {0}".format(weights_filename))

    arch_filename = "%s_arch.json" % model_name
    model_arch = model.to_json()
    with open( arch_filename, 'w' ) as arch_file:
        arch_file.write( model_arch )
    print ("Saved model architecture to {0}".format(arch_filename))
