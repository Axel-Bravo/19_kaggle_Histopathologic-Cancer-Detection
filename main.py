#%% Imports and functions
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% Data pre-processing
train_dir = "data/train/"
test_dir = "data/test/"

# Parameters
batch_size = 128
input_dimension = (94, 94)
kernel_size = (3, 3)
pool_size = (2, 2)
first_filters = 32
second_filters = 64
third_filters = 128
dropout_conv = 0.5
dropout_dense = 0.6

model_name = 'model_20'

# Train/Val
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_dimension,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='binary',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_dimension,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')

# Test
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_dimension,
    color_mode="rgb",
    shuffle=False,
    batch_size=2)


# %% Model - Initialization

# CNNs I
model = tf.keras.Sequential()
model.add(layers.Conv2D(input_shape=(*input_dimension, 3), filters=first_filters, kernel_size=kernel_size, strides=1))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(input_shape=(*input_dimension, 3), filters=first_filters, kernel_size=kernel_size, strides=1))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Conv2D(filters=first_filters, kernel_size=kernel_size, strides=1))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPool2D(pool_size=pool_size))
# CNNs II
model.add(layers.Conv2D(filters=second_filters, kernel_size=kernel_size, strides=1))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=second_filters, kernel_size=kernel_size, strides=1))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Dropout(dropout_conv))
model.add(layers.Conv2D(filters=second_filters, kernel_size=kernel_size, strides=1))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Dropout(dropout_conv))
model.add(layers.MaxPool2D(pool_size=pool_size))
# CNNs III
model.add(layers.Conv2D(filters=third_filters, kernel_size=kernel_size, strides=1))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=third_filters, kernel_size=kernel_size, strides=1))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Dropout(dropout_conv))
model.add(layers.Conv2D(filters=third_filters, kernel_size=kernel_size, strides=1))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Dropout(dropout_conv))
model.add(layers.MaxPool2D(pool_size=pool_size))
# Dense
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(dropout_dense))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(dropout_dense))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# Compile
optimizer = optimizers.RMSprop(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# %% Model - Training
n_epochs = 50
data_augmentation_coef = 1.0

# Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.025, patience=10, restore_best_weights=True,
                                     verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', min_delta=0.025, patience=5, min_lr=0.00001,
                                        factor=0.5, verbose=1)
model_checker = callbacks.ModelCheckpoint(filepath='models/' + model_name, monitor='val_accuracy', save_best_only=True,
                                          save_weights_only=True, verbose=1)
tensorboard = callbacks.TensorBoard(log_dir='logs/' + model_name)  # tensorboard --logdir=logs/model_20/

model.fit_generator(train_generator, steps_per_epoch=train_generator.samples * data_augmentation_coef // batch_size,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // batch_size,
                    epochs=n_epochs,
                    callbacks=[early_stop, reduce_lr, model_checker, tensorboard],
                    workers=15,
                    use_multiprocessing=True)


# %% Model - Predict
predictions = model.predict_generator(test_generator, steps=test_generator.samples//2, verbose=1,
                                      workers=15, use_multiprocessing=True)


# %% Predictions - Post-processing
filenames = test_generator.filenames
filenames = [file.split(sep='/')[1].split(sep='.')[0] for file in filenames]

results = pd.DataFrame({"id": filenames,
                        "label": predictions.ravel()})

results.to_csv('submissions/' + model_name + '.csv', index=False)

# %% Model - Kaggle evaluation
subprocess.run('kaggle competitions submit -c histopathologic-cancer-detection -f submissions/' + model_name + '.csv'
               ' -m ' + '"' + model_name + '"', shell=True)
