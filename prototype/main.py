# This is a sample Python script.

# Custom modules
import constants

# Public modules
import os
import pathlib
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input


def load_and_resize_image(img_path):
    """
    This method loads the image

    :param img_path: path to the image to process
    :return: resized images
    """

    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_raw, channels=constants.CHANNELS)
    img = tf.image.resize(img, constants.IMG_SIZE)
    return img


def get_file_paths_and_labels(data_root):
    """
    Method returns a list of paths to images, corresponding
    class labels and mapping of class names to label index based on
    the provided data root directory.

    Note! Currently, supports only JPEG files!
    """

    image_paths = sorted([str(path) for path in data_root.glob('*/*.jpg')])
    random.shuffle(image_paths)

    label_names = extract_label_names(data_root)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    labels = [label_to_index[pathlib.Path(path).parent.name] for path in image_paths]

    return image_paths, labels, label_to_index


def extract_label_names(data_root):
    return sorted(item.name for item in data_root.glob('*/') if item.is_dir())


def draw_document_samples(image_paths, labels, label_to_index):
    plt.figure(figsize=(25, 15))
    for i, (label_name, label_int) in enumerate(label_to_index.items()):
        sample_idx = labels.index(label_int)
        image_path = image_paths[sample_idx]
        img = Image.open(image_path).convert('1')
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(True)
        plt.imshow(img)
        plt.xlabel(label_name)
    plt.show()


def draw_count_by_category(labels,
                           label_to_index):
    plt.figure(figsize=(12, 6))
    plt.hist(labels, bins=range(len(label_to_index) + 1))
    ax = plt.gca()
    ax.set_title('Label distribution of document dataset per class')
    ax.set_xticks([i + 0.5 for i in range(len(label_to_index))])
    _ = ax.set_xticklabels(list(label_to_index.keys()))
    plt.xticks(rotation=10, ha='right')
    plt.show()


def draw_training_history(history):
    train_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    train_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([min(plt.ylim()), 1])
    plt.xticks(range(len(train_accuracy)))
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.xticks(range(len(train_loss)))

    plt.tight_layout()
    plt.show()


def draw_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = title
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


def zip_datasets(images, labels):
    """
    Returns a zipped tf.data.Dataset from the two input datasets where first one defines images and second one defiles
    the class labels.
    """
    return tf.data.Dataset.zip(images, labels)


def create_input_pipeline(image_paths, labels,
                          batch_size=constants.BATCH_SIZE, is_training=False):
    """
    Create a tf.data.Dataset pipeline from the given file paths and labels.
    The pipeline includes image preprocessing, batching and creates a zipped
    dataset that feeds both images and text to the input layer.
    If `is_train==True` the dataset also gets shuffle and repeated for being
    used as training input.
    """
    image_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = image_ds.map(load_and_resize_image).map(preprocess_input)

    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    ds = zip_datasets(image_ds, label_ds)

    if is_training:
        ds = ds.shuffle(buffer_size=len(image_paths)).repeat()

    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def build_vgg16_tfidf_model():
    """
    Builds a keras.Model consisting of a pretrained VGG16 network.
    """

    # Image path: using a VGG16 network, pretrained on ImageNet
    vgg16_model = tf.keras.applications.VGG16(
        input_shape=[constants.IMG_SIZE[0], constants.IMG_SIZE[1], constants.CHANNELS],
        include_top=False, weights='imagenet')
    vgg16_model.trainable = True

    image_input = tf.keras.layers.Input(shape=(constants.IMG_SIZE[0], constants.IMG_SIZE[1], constants.CHANNELS))
    image_model = vgg16_model(image_input)
    # The output of the image path are the averaged values of each convolutional
    # filter in the top layer
    image_model = tf.keras.layers.GlobalAveragePooling2D()(image_model)

    batch_normalization_layer = tf.keras.layers.BatchNormalization(momentum=0.9)(image_model)
    output = tf.keras.layers.Dense(constants.NUM_LABELS, activation='softmax')(batch_normalization_layer)

    model = tf.keras.Model(inputs=[image_input], outputs=output)

    return model


def train_model(model_name):
    print('Version of terraform used: %s' % tf.__version__)

    print('===   Start processing')

    # Initializing randoms
    # TODO Add more description here.
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    # Check files in the dataset that will be used for Machine Learning
    data_root = pathlib.Path(constants.ORIGINAL_DATASET_PATH)

    print('===   List of folders in the dataset')
    for item in data_root.iterdir():
        print(item)

    # Load image paths, labels and label to index mapping from the dataset.
    # TODO get labels amount dynamically instead of doing this statically.
    image_paths, labels, label_to_index = get_file_paths_and_labels(data_root)
    # TODO enable drawing of the documents.
    # draw_document_samples(image_paths, labels, label_to_index)
    print('===   Amount of images in the dataset: %s' % len(image_paths))
    # draw_cound_by_category(labels, label_to_index)

    # Creating the training, validation and testing datasets for the original dataset.
    ds_train = create_input_pipeline(image_paths[:constants.TRAIN_SIZE],
                                     labels[:constants.TRAIN_SIZE], is_training=True)
    ds_val = create_input_pipeline(image_paths[constants.TRAIN_SIZE: constants.TRAIN_SIZE + constants.VAL_SIZE],
                                   labels[constants.TRAIN_SIZE: constants.TRAIN_SIZE + constants.VAL_SIZE],
                                   is_training=False)
    ds_test = create_input_pipeline(image_paths[constants.TRAIN_SIZE + constants.VAL_SIZE:],
                                    labels[constants.TRAIN_SIZE + constants.VAL_SIZE:],
                                    is_training=False)

    # Store the correct values of labels for test set. Will be used later for testing of the predictions.
    expected_correct_test_labels = labels[constants.TRAIN_SIZE + constants.VAL_SIZE:]

    # Build a model to be trained and print its summary
    model = build_vgg16_tfidf_model()
    print('===   Model summary')
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Calculate the number of steps per epoch that will be used during model training.
    steps_per_epoch = np.ceil(constants.TRAIN_SIZE / constants.BATCH_SIZE)
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

    # Visualising the result of the training
    draw_training_history(history)

    test_loss, test_accuracy = model.evaluate(ds_test)
    print(f"===   Test accuracy: {test_accuracy}")

    predictions = model.predict(ds_test)
    predicted_labels = np.argmax(predictions, axis=1)
    print(
        classification_report(expected_correct_test_labels, predicted_labels, target_names=list(label_to_index.keys()),
                              digits=4))

    draw_confusion_matrix(expected_correct_test_labels, predicted_labels, classes=list(label_to_index.keys()),
                          normalize=True, title="Evaluation on the test subset from original dataset.")

    print('===   Storing the model')
    model.save(model_name)
    return model


def classify_documents(source_dir, model):
    image_paths = [str(path) for path in pathlib.Path(source_dir).glob('*.jpg')]
    print('===   Images to classify %s' % image_paths)

    dataset_to_classify = create_input_pipeline(
        np.array(image_paths),
        np.array(['Empty'] * len(image_paths)), len(image_paths), False)

    predictions = model.predict(dataset_to_classify)

    label_names = extract_label_names(pathlib.Path(constants.ORIGINAL_DATASET_PATH))

    predicted_indexes = np.argmax(predictions, axis=1)
    print('===   Predicted labels: ')
    for index in predicted_indexes:
        print(label_names[index])


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

    if use_stored_model:

        print('===   Loading stored model')
        model = tf.keras.models.load_model(constants.STORED_MODEL_FILE)
        model.summary()
    else:
        print('===   Training new model')
        model = train_model(constants.STORED_MODEL_FILE)

    location_of_files_to_read = input('Please, specify the folders with files to verify model work: ')
    classify_documents(location_of_files_to_read, model)
