# This is a main entry point of the program

# Custom modules
import constants
import drawings
import models

# Public modules
import os
import pathlib
import random
import shutil
import zipfile

import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf


def __get_file_paths_and_labels(data_root):
    """
    Method returns a list of paths to images, corresponding
    class labels and mapping of class names to label index based on
    the provided data root directory.

    Note! Currently, supports only JPEG files!
    """

    image_paths = sorted([str(path) for path in data_root.glob('*/*.jpg')])
    random.shuffle(image_paths)

    label_names = __extract_label_names(data_root)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    labels = [label_to_index[pathlib.Path(path).parent.name] for path in image_paths]

    return image_paths, labels, label_to_index


def __extract_label_names(data_root):
    """
    Extracts sorted list of label names from original dataset.
    """

    return sorted(item.name for item in data_root.glob('*/') if item.is_dir())


def train_model(datasets_path, dataset_name):
    """
    Trains the new model from the beginning.

    :param datasets_path: the path to the folder where all datasets are located
    :param dataset_name: the name of the folder with dataset to use
    :return: trained model
    """

    print('Version of terraform used: %s' % tf.__version__)

    # Reset the seeds so the random numbers will be the same every time.
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    # check whether extracted dataset from archive exists, if no - extract the files for further processing
    dataset_folder = os.path.join(datasets_path, dataset_name)
    is_dataset_folder_existed = os.path.exists(dataset_folder)
    if not is_dataset_folder_existed:
        print("===   No extracted files found, so start the process from extraction from archive")
        with zipfile.ZipFile(os.path.join(dataset_folder + '.zip')) as archive:
            for file in archive.namelist():
                archive.extract(file, datasets_path)

    print('===   Start training process')
    # Check files in the dataset that will be used for Machine Learning
    data_root = pathlib.Path(dataset_folder)

    print('===   List of folders in the dataset')
    for item in data_root.iterdir():
        print(item)

    # Load image paths, labels and label to index mapping from the dataset.
    image_paths, labels, label_to_index = __get_file_paths_and_labels(data_root)
    # Draw sample of the documents and distribution of the documents by classes.
    drawings.draw_document_samples(image_paths, labels, label_to_index)
    drawings.draw_distribution_of_documents_per_class(labels, label_to_index)

    # Calculate train, validation and test subsets size based on the preconfigured ratio
    total_amount_of_images = len(image_paths)
    train_size = int(total_amount_of_images * constants.TRAIN_RATIO)
    validation_size = int(total_amount_of_images * constants.VALIDATION_RATIO)
    test_size = total_amount_of_images - train_size - validation_size
    print('===   Total amount of images in the dataset: %s' % total_amount_of_images)
    print('===   The size of the training subset: %s' % train_size)
    print('===   The size of the validation subset: %s' % validation_size)
    print('===   The size of the testing subset: %s' % test_size)

    # Creating the training, validation and testing datasets for the original dataset.
    # Includes image preprocessing by resizing and applying VGG16 standard preprocessing.
    ds_train = models.create_input_pipeline(image_paths[:train_size],
                                            labels[:train_size], is_training=True)
    ds_val = models.create_input_pipeline(image_paths[train_size: train_size + validation_size],
                                          labels[train_size: train_size + validation_size],
                                          is_training=False)
    ds_test = models.create_input_pipeline(image_paths[train_size + validation_size:],
                                           labels[train_size + validation_size:],
                                           is_training=False)

    # Store the correct values of labels for test set. Will be used later for testing of the predictions.
    expected_correct_test_labels = labels[train_size + validation_size:]

    # Build a model to be trained and print its summary
    model = models.build_vgg16_based_model(len(label_to_index.keys()))
    print('===   Model summary')
    model.summary()

    # Compile the model with Adam optimizer, a popular optimization algorithm known for its efficiency in deep
    # learning tasks. The Adam optimizer adapts the learning rates for different parameters dynamically,
    # thereby accelerating the convergence of the network. The model is trained using the standard cross-entropy loss
    # function.
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Calculate the number of steps per epoch that will be used during model training based on the training array size
    # and batch size
    steps_per_epoch = np.ceil(train_size / constants.BATCH_SIZE)
    print('===   Calculated steps per epoch: %s' % steps_per_epoch)

    # Start training of the model.
    print('===   Start training of the model')
    history = model.fit(ds_train, epochs=100,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=ds_val,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor='val_accuracy', min_delta=0, patience=5,
                            restore_best_weights=True)],
                        workers=12)

    # Visualising the result of the training as a separate plot.
    drawings.draw_training_history(history)

    test_loss, test_accuracy = model.evaluate(ds_test)
    print(f"===   Test accuracy: {test_accuracy}")

    # Verify the model based on the test dataset by making the predictions and building classification report and
    # drawing the confusion matrix.
    predictions = model.predict(ds_test)
    predicted_labels = np.argmax(predictions, axis=1)
    print(
        classification_report(expected_correct_test_labels, predicted_labels, target_names=list(label_to_index.keys()),
                              digits=4))

    drawings.draw_confusion_matrix(expected_correct_test_labels, predicted_labels, classes=list(label_to_index.keys()),
                                   normalize=True, title="Evaluation on the test subset from original dataset.")

    return model


def classify_documents(source_dir, trained_model):
    """
    Method classifies the documents from the supplied source directory using the trained model and move them
    to the pre-configured static output folder. Each document will be moved to the specific sub-folder that
    corresponds to the predicted class.

    :param source_dir: the directory from which the files will be loaded and classified
    :param trained_model: the pre-trained model that will be used for classification
    :return: None
    """

    image_paths = [str(path) for path in pathlib.Path(source_dir).glob('*.jpg')]

    # Additional check to prevent processing of non-existed or empty folders.
    if len(image_paths) == 0:
        print("===   No images found in the folder, either the folder do not exist or it is empty!")
        return

    print('===   Images to classify %s' % image_paths)

    # Create an empty label array just to reuse existed function for building dataset.
    # It will not affect predictions and will not be used there.
    dataset_to_classify = models.create_input_pipeline(
        np.array(image_paths),
        np.array(['Empty'] * len(image_paths)), len(image_paths), False)

    predictions = trained_model.predict(dataset_to_classify)

    # Extract label names from initial dataset. Keras and Tensorflow do not allow to extract label names
    # from the stored model.
    label_names = __extract_label_names(pathlib.Path(os.path.join(constants.DATASETS_PATH, constants.DATASET_TO_TRAIN)))

    # Print the predicted labels for each document to console and move file to the corresponding folder.
    normalized_predictions = np.argmax(predictions, axis=1)
    print('===   Predicted labels: ')
    for index, prediction in enumerate(normalized_predictions):
        predicted_label = label_names[prediction]
        current_image = image_paths[index]
        print(f"*   Image: {current_image}; predicted as {predicted_label}")
        target_directory = os.path.join(constants.CLASSIFIED_DOCUMENTS_DIRECTORY, predicted_label)

        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        shutil.move(current_image, target_directory)


# The main entry of the program
if __name__ == '__main__':
    # Check whether trained model is stored previously
    is_store_model_existed = os.path.exists(constants.STORED_MODEL_FILE)
    use_stored_model = False

    # If stored model exists - ask user whether he wants to use it
    if is_store_model_existed:
        input_for_question = input('There is a stored model existed, would you like to use it? Print yes to proceed '
                                   'with it, any other answer will be treated as no: ')
        use_stored_model = input_for_question == 'yes'

    # If user wants to use stored model, then load it and print its summary to logs.
    if use_stored_model:
        print('===   Loading stored model')
        model = tf.keras.models.load_model(constants.STORED_MODEL_FILE)
        model.summary()
    # If no, start the process of model training from the beginning
    else:
        print('===   Training new model')
        model = train_model(constants.DATASETS_PATH, constants.DATASET_TO_TRAIN)
        print('===   Storing the trained model')
        model.save(constants.STORED_MODEL_FILE)

    # Provide a chance for the user to classify additional documents from the particular folder.
    location_of_files_to_read = input('Please, specify the absolute path to the folders with files to classify: ')
    classify_documents(location_of_files_to_read, model)
