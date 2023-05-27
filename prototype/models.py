import constants

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input


def __load_and_resize_image(img_path):
    """
    This method loads the image by its path and resize it to predefined size.

    :param img_path: path to the image to process
    :return: resized images
    """

    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_raw, channels=constants.CHANNELS)
    img = tf.image.resize(img, constants.IMG_SIZE)
    return img


def __zip_datasets(images, labels):
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
    image_ds = image_ds.map(__load_and_resize_image).map(preprocess_input)

    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    ds = __zip_datasets(image_ds, label_ds)

    if is_training:
        ds = ds.shuffle(buffer_size=len(image_paths)).repeat()

    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def build_vgg16_based_model():
    """
    Builds a keras.Model consisting of a pretrained VGG16 network with additional Input layer before it and multiple
    additional output layers after it - GlobalAveragePooling2D, BatchNormalization and Dense.

    :return: created CNN model
    """

    # Using a VGG16 network, pretrained on ImageNet
    vgg16_model = tf.keras.applications.VGG16(
        input_shape=[constants.IMG_SIZE[0], constants.IMG_SIZE[1], constants.CHANNELS],
        include_top=False, weights='imagenet')
    vgg16_model.trainable = True

    image_input = tf.keras.layers.Input(shape=(constants.IMG_SIZE[0], constants.IMG_SIZE[1], constants.CHANNELS))
    image_model = vgg16_model(image_input)
    # The output of the image path are the averaged values of each convolutional
    # filter in the top layer
    image_model = tf.keras.layers.GlobalAveragePooling2D()(image_model)

    # At the end apply batch normalization and dense layers
    batch_normalization_layer = tf.keras.layers.BatchNormalization(momentum=0.9)(image_model)
    output = tf.keras.layers.Dense(constants.NUM_LABELS, activation='softmax')(batch_normalization_layer)

    model = tf.keras.Model(inputs=[image_input], outputs=output)

    return model
