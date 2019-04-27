# %% 0| Import and function declaration
import random
import pathlib
import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


# Image processing
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [90, 90])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


# %% 1| Data Load
# 1.1| X_train data
# Create images paths
X_data_path = pathlib.Path('data/train/')
X_data_paths = list(X_data_path.glob('*.tif'))
X_data_paths = [str(path) for path in X_data_paths]
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


# %%

img_raw = tf.io.read_file(X_data_paths[1])

# %%

X_data_paths[:10]
