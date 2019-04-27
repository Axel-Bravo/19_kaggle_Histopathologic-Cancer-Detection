# %% 0| Import and function declaration
import pathlib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, PReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

AUTOTUNE = tf.data.experimental.AUTOTUNE


# Image processing
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)  # TODO: change file format to jpeg
    image = tf.image.resize(image, [90, 90])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


class PRELU(PReLU):
    def __init__(self, **kwargs):
        self.__name__ = "PRELU"
        super(PRELU, self).__init__(**kwargs)


# %% 1| Data Load
# 1.1| X_train data
# Create images paths
X_data_path = pathlib.Path('data/train/')
X_data_paths = list(X_data_path.glob('*.tif'))
X_data_paths = [str(path) for path in X_data_paths]
len_data = len(X_data_paths)
X_train_paths = X_data_paths[22002:]
X_val_paths = X_data_paths[: 22002]
# Create Tensorflow dataset
X_train_path_ds = tf.data.Dataset.from_tensor_slices(X_train_paths)
X_train_ds = X_train_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
X_val_path_ds = tf.data.Dataset.from_tensor_slices(X_val_paths)
X_val_ds = X_val_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

del X_data_path, X_train_paths, X_val_paths, X_train_path_ds, X_val_path_ds

# 1.2| y_train data
# Create images paths
Xy_data_mapper = pd.read_csv('data/train_labels.csv')
X_data_order = pd.DataFrame([pict_id.split(sep='/')[2].split(sep='.')[0] for pict_id in X_data_paths])
y_data = list(X_data_order.merge(Xy_data_mapper, how='inner', left_on=0, right_on='id')['label'])
y_train = y_data[22002:]
y_val = y_data[: 22002]
# Create Tensorflow dataset
y_train_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y_train, tf.int64))
y_val_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y_val, tf.int64))

del Xy_data_mapper, X_data_paths, X_data_order, y_data, y_train, y_val

# 1.3| train_data & val_data
train_ds = tf.data.Dataset.zip((X_train_ds, y_train_ds))
val_ds = tf.data.Dataset.zip((X_val_ds, y_val_ds))

del X_train_ds, y_train_ds, X_val_ds, y_val_ds


# %% Data Training preparation
BATCH_SIZE = 128
train_ds = train_ds.shuffle(buffer_size=15000).repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.shuffle(buffer_size=1500).repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)


# %% Model Creation
model = Sequential()
model.add(Conv2D(input_shape=(90, 90, 3), filters=64, kernel_size=(4, 4), strides=(2, 2),
                 activation=PRELU(),
                 kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.4, l2=0.4)))
model.add(MaxPool2D())
model.add(Conv2D(input_shape=(90, 90, 3), filters=64, kernel_size=(4, 4), strides=(2, 2),
                 activation=PRELU(),
                 kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.4, l2=0.4)))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# %% Model Training

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=["accuracy"])

train_steps = len_data // BATCH_SIZE

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.05, patience=4, verbose=1)
model_check = ModelCheckpoint('models/', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=3, min_lr=0.001)
tensorboard = TensorBoard()

model.fit(train_ds, epochs=100, steps_per_epoch=train_steps,
          validation_data=val_ds,
          callbacks=[early_stop, model_check, reduce_lr, tensorboard])
