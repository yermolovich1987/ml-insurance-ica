# This file contain the code responsible for drawing plots.

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix


def draw_document_samples(image_paths, labels, label_to_index):
    """
    This function prints and plots the document samples in a corresponding grid.
    The plot contain one sample document per class.
    """

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


def draw_distribution_of_documents_per_class(labels,
                                             label_to_index):
    """
    This function prints and plots the distribution of document dataset per class (number of documents per a class in
    the dataset)
    """

    plt.figure(figsize=(12, 6))
    plt.hist(labels, bins=range(len(label_to_index) + 1))
    ax = plt.gca()
    ax.set_title('Distribution of document dataset per class')
    ax.set_xticks([i + 0.5 for i in range(len(label_to_index))])
    _ = ax.set_xticklabels(list(label_to_index.keys()))
    plt.xticks(rotation=10, ha='right')
    plt.show()


def draw_training_history(history):
    """
    This function draws the training history result.
    """

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
