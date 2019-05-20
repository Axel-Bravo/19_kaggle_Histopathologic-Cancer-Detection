#%% Imports and functions
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers

#%% Load Best model

model = models.load_model('filepath')
model.summary()

#%%
