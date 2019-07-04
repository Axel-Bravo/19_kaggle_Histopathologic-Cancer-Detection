#%% Imports and functions
import tensorflow as tf
import matplotlib.pyplot as plt
from vis.visualization import visualize_activation
from vis.utils import utils
from tensorflow.keras import activations
from tensorflow.keras import applications
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform




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

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = tf.keras.models.load_model('models/model_21__32-0.15.hdf5')
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
plt.rcParams['figure.figsize'] = (18,6)

#finding out the layer index using layer name
#the find_layer_idx function accepts the model and name of layer as parameters and return the index of respective layer
layer_idx = utils.find_layer_idx(model, 'dense_3')
#changing the activation of the layer to linear
model.layers[layer_idx].activation = activations.linear
#applying modifications to the model
model = utils.apply_modifications(model)
#Indian elephant
img3 = visualize_activation(model,layer_idx,filter_indices=1,max_iter=5000,verbose=True)
plt.imshow(img3)