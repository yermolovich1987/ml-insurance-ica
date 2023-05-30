Add a dataset to this folder and configure the constant DATASET_TO_TRAIN with the name of the folder containing dataset or 
with the name of zip archive that contains it (without extension). In later case application will automatically unzip
the archive.

The dataset should contain images sorted by folders that represents different document classes. With the top level folder
representing the dataset name mentioned in the constant DATASET_TO_TRAIN.

By default, application preconfigured to work with the dataset named "claims_for_ica"

Sample of the dataset layout:

![sample_of_dataset_layout.png](..%2Fsample_of_dataset_layout.png)