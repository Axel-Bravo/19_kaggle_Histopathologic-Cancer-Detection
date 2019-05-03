#%% Import and function declaration
import os
import shutil
import pandas as pd


def file_processor(source_folder: str, target_folder: str) -> None:
    """
    Generates the folder structure necesary to use the "Keras - flow from diractory" method
    :param source_folder: original folder where the images are
    :param target_folder: forlder where the images will be created
    :return: None
    """
    dataset = pd.read_csv(source_folder + "train_labels.csv")
    file_names = list(dataset['id'].values)
    img_labels = list(dataset['label'].values)

    for label in set(img_labels):
        os.makedirs(target_folder + str(label) + '/')

    for f in range(len(file_names)):
        current_img = file_names[f]
        current_label = img_labels[f]
        shutil.move(source_folder + "train/" + current_img + '.tif',
                    target_folder + str(current_label) + "/" + current_img + '.tif')


if __name__ == '__main__':
    file_processor(source_folder='data/', target_folder='data/train_type')
