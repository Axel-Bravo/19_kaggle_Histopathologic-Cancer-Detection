#%% Imports and functions
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#%% Data pre-processing
train_dir ="data/train_type/"

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        horizontal_flip=True,
        validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(32, 32),
        color_mode="rgb",
        batch_size=64,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(32, 32),
        color_mode="rgb",
        batch_size=64,
        class_mode='categorical',
        subset='validation')

#%% Model - Definition
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#%% Model - Training
n_epochs = 20
batch_size = 128

early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=4, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', min_delta=0.05, patience=2, min_lr=0.001,
                                        factor=0.5, verbose=1)
model_checker = callbacks.ModelCheckpoint(filepath='models/', monitor='val_accuracy', save_best_only=True,
                                save_weights_only=True)
tensorboard = callbacks.TensorBoard(log_dir='logs/')  # tensorboard --logdir=logs/

model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // batch_size,
                    epochs=n_epochs,
                    callbacks=[early_stop, reduce_lr, model_checker, tensorboard],
                    workers=15,
                    use_multiprocessing=True)

