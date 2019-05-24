#%% Imports and functions
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers


def get_layers_info(model) -> dict:
    """
    Create a dictionary layers_info which maps a layer name to its charcteristics
    :param model: Keras' Sequential Model
    :return: dictionary containing layer's information
    """
    layers_info = {}
    for i in model.layers:
        layers_info[i.name] = i.get_config()

    return layers_info


def get_layers_weights(model) -> dict:
    """
    Create a dictionary of the layer_weights dictionary will map every layer_name to its corresponding weights
    :param model: Keras' Sequential Model
    :return: dictionary containing layer's weights
    """
    layer_weights = {}
    for i in model.layers:
        layer_weights[i.name] = i.get_weights()

    return layer_weights


#%% Load Best model
model = models.load_model('models/model_21__32-0.15.hdf5')
model.summary()

layers_info = get_layers_weights(model=model)
layers_weights = get_layers_weights(model=model)
layers = model.layers


#%% Plot some filters
layer = 'conv2d'
fig, ax = plt.subplots(nrows=1, ncols=5)

for i in range(5):
    ax[i].imshow(layers_weights[layer][0][:, :, :, i][:, :, 0],
                 cmap='gray')
    ax[i].set_title('Filter: '+str(i+1))
    ax[i].set_xticks([])
ax[i].set_yticks([])
fig.suptitle("Filters of layer {}".format(layer))
plt.show()


#%% Activation Maximization

