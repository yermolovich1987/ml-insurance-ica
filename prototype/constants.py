BATCH_SIZE = 16
# Size of the image.
IMG_SIZE = [224, 224]
CHANNELS = 3
NUM_LABELS = 7
# The size of the training dataset that will be used
TRAIN_SIZE = 1000
# The size of the validation dataset that will be used
VAL_SIZE = 500

# The path to the dataset that will be used for model training.
ORIGINAL_DATASET_PATH = './datasets/claims_for_ica/train'
# Preconfigured path where pre-trained model will be stored
STORED_MODEL_FILE = './saved_models/trainedModel.h5'
# Folder where the additional classified documents will be stored
CLASSIFIED_DOCUMENTS_DIRECTORY = "./classified_documents/"
