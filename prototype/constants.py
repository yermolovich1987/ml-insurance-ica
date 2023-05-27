BATCH_SIZE = 16
# Size of the image.
IMG_SIZE = [224, 224]
CHANNELS = 3
NUM_LABELS = 7
# The size of the training dataset that will be used
TRAIN_SIZE = 1000
# The size of the validation dataset that will be used
VAL_SIZE = 500

# Path to the folder with datasets
DATASETS_PATH = './datasets'
# The name of the dataset that will be used for testing
DATASET_TO_TRAIN = 'claims_for_ica'
# Preconfigured path where pre-trained model will be stored
STORED_MODEL_FILE = './saved_models/trainedModel.h5'
# Folder where the additional classified documents will be stored
CLASSIFIED_DOCUMENTS_DIRECTORY = "./classified_documents"
