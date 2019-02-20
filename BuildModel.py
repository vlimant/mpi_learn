### Builds one of the available models.  
# Saves model architecture to <model_name>_arch.json
# and model weights to <model_name>_weights.h5

import os
os.environ['CUDA_VISIBLE_DEVICES']=""
import argparse

from models.Models import make_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='model to construct')
    parser.add_argument('model_args', nargs='*', help='key=value to pass to the model',default=[])
    args = parser.parse_args()
    model_name = args.model_name
    model_args = {}
    for kw in args.model_args:
        k,v = kw.split('=')
        try:
            v = int(v)
        except:
            v= float(v)
        model_args[k] = v
    if model_args:
        print ("passing",model_args,"to the model builder")
        model = make_model( model_name ,**model_args)
    else:
        model = make_model( model_name)
    weights_filename = "%s_weights.h5" % model_name
    arch_filename = "%s_arch.json" % model_name

    if not "torch" in model_name:
        model.summary()
        model.save_weights( weights_filename, overwrite=True )
        print ("Saved model weights to {0}".format(weights_filename))

        model_arch = model.to_json()
        with open( arch_filename, 'w' ) as arch_file:
            arch_file.write( model_arch )
        print ("Saved model architecture to {0}".format(arch_filename))
    else:
        import torch
        weights_filename = weights_filename.replace('h5','torch')
        arch_filename = arch_filename.replace('json','torch')
        torch.save(model.state_dict(), weights_filename)
        print ("Saved model weights to {0}".format(weights_filename))
        torch.save(model, arch_filename)
        print ("Saved model architecture to {0}".format(arch_filename))
                        
