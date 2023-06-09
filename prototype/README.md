# Introduction

This project represents a prototype for work on improving insurance
document processing flow by automation of insurance document classification
applying Machine Learning, in particular - deep neural networks.

# Requirements

### Required dependencies

To run prototype you need to install the required modules (for example, using pip), in particular:

* tensorflow version 2.13.0rc0 or greater (note that there is a bug in the tensorflow 2.12.* that's why Dataset.zip function will not work properly).
* scikit-learn version 1.2.2 or greater.
* numpy version 1.24.3 or greater.
* matplotlib version 3.7.1 or greater

This versions were tested both on MacBook Pro M1 and on old Windows machine.

### Dataset structure and location.

The application expects the dataset represented by document images in jpeg format
pre-classified by different folders. Name of the folder describes the class to be
predicted. The folder where dataset should be located is ```.../prototype/datasets```.
More requirements to the dataset could be found in:
https://github.com/yermolovich1987/ml-insurance-ica/blob/main/prototype/datasets/README.md

The dataset that was used to test the program (located in claims_for_ica.zip) is taken from the Roboflow site and then
modified manually to suit the needs of this work. In particular, original classification where changed
and documents were moved to the folders that describes the document classes. The dataset is not supplied with 
the code due to the big size of archive. 

The dataset archive could be downloaded via the link:
TODO Add link

Reference to the original dataset:

```json
@misc{
  claims_dataset,
  title
  = {
  Claims
  Dataset
},
type = {Open Source Dataset},
author = {Hobby},
howpublished = {\url{ https: //universe.roboflow.com/hobby-nxqek/claims } },
url = {https: //universe.roboflow.com/hobby-nxqek/claims },
journal = { Roboflow Universe},
publisher = {Roboflow},
year = {2022},
month = {oct },
note = {visited on 2023-05-26},
}
```

# Running of the application

The entry point of the application is ```main.py``` file. Application could be executed rather through the IDE or from
terminal by running:

```bash
python3 main.py
```

_**Note***_ During execution of the program it will print comprehensive logs about execution
process and plot the images with the document samples or statistic. If the image window
is shown then to continue execution of the program you will need to close the window.

After the training, the model is stored in the h5 format on disk (newer keras
format is not used since it cause problems after loading of the module on the Apple M1 chip).
If the trained model already exists, then the application will ask whether you would like to
use a stored trained model, or train new model from the beginning.

At the end of the running, the application will ask the location of the folder with custom
document images to classify. This could be useful for additional testing on custom images,
or if the user wants to use already trained model to classify the documents simulating
the real environment. The document in this case will be moved to the proper sub-folder that corresponds
to the document clas inside the ```.../prototype/classified_documents``` folder (the name of this folder could be changed 
through CLASSIFIED_DOCUMENTS_DIRECTORY constant).

The archive with the sample of the files to run classification after model is trained and tested, could be downloaded via the link below.
This document images was manually edited from the original dataset to change their content, or generated from scratch.
TODO Add link

The location of the dataset, location where to store trained model and other important training
parameters are configured through the constants located in ```constants.py``` module.

The training process on the included dataset takes quite a long time (on powerful machine using GPU about 3-5 minutes)

# Limitations

* Right now this prototype works only with jpeg document images (*.jpg file extension).