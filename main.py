# %% Imports and functions
import subprocess
import numpy as np
import pandas as pd
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% Data pre-processing
train_dir = "data/train/"
test_dir = "data/test/"

# Parameters
batch_size = 64
input_dimension = (50, 50)
model_name = 'model_06'

# Train/Val
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    shear_range=10,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_dimension,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_dimension,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# Test
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_dimension,
    color_mode="rgb",
    shuffle=False,
    batch_size=2)

#test_generator.reset()

# %% Model - Initialization

l2_reg = 0.01

# Definition
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3),))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l=l2_reg)))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l=l2_reg)))
model.add(layers.BatchNormalization())
model.add(layers.Dense(12, activation='relu', kernel_regularizer=regularizers.l2(l=l2_reg)))
model.add(layers.Dense(2, activation='softmax'))
model.summary()

# Compile
optimizer = optimizers.RMSprop(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# %% Model - Training
n_epochs = 50
data_augmentation_coef = 1.0

# Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.025, patience=10, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', min_delta=0.025, patience=5, min_lr=0.001,
                                        factor=0.5, verbose=1)
model_checker = callbacks.ModelCheckpoint(filepath='models/' + model_name, monitor='val_accuracy', save_best_only=True,
                                          save_weights_only=True, verbose=1)
tensorboard = callbacks.TensorBoard(log_dir='logs/' + model_name)  # tensorboard --logdir=logs/model_06/

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
predicted_class = np.argmax(predictions, axis=1)
filenames = test_generator.filenames
filenames = [file.split(sep='/')[1].split(sep='.')[0] for file in filenames]

results = pd.DataFrame({"id": filenames,
                        "label": predicted_class})

results.to_csv('submissions/' + model_name + '.csv', index=False)

# %% Model - Kaggle evaluation
subprocess.run('kaggle competitions submit -c histopathologic-cancer-detection -f submissions/' + model_name + '.csv'
               ' -m ' + '"' + model_name + '"', shell=True)
