### Predefined Keras models

from keras.models import Sequential
from keras.layers import Dense, Activation

def make_model(model_name):
    """Constructs the Keras model indicated by model_name"""
    return model_maker_dict[model_name]()

def make_example_model():
    """Example model from keras documentation"""
    model = Sequential()
    model.add(Dense(output_dim=64, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

model_maker_dict = {
        'example':make_example_model,
        }
