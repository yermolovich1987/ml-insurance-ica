# Introduction
This project represents a prototype for work of improving insurance
document processing flow by automation of insurance document classification
by applying Machine Learning, in particular - deep neural networks.

# Requirements

### Required dependencies
To run prototype you need to install the required modules (for example, using pip), in particular:
* tensorflow version 2.13.0 or greater.
* scikit-learn version 1.2.2 or greater.
* numpy version 1.24.3 or greater.

### Dataset structure and location.

The application expects the dataset represented by document images in jpeg format
pre-classified by different folders. Name of the folder describes the class to be 
predicted.

# Running of the application
The application could be executed rather through the IDE or from a simple console:

```bash
python3 main.py
```

_**Note***_ During execution of the program it will print comprehensive logs about execution 
process and plot the images with the document samples or statistic. If image window
is shown then to continue execution you will need to close 

At the end of training the pretrained model is stored in the h5 format on disk (newer keras 
format is not used since it cause problems after loading of the module on the Apple M1 chip). 
If the trained model already exists, then the application will ask whether you would like to
use a stored trained model, or train new model from the beginning.

At the end of the running, the application will ask the location of the folder with
document to classify.

The location of the dataset, location where to store trained model and other important training
parameters are configured through the constants located in ```constants.py``` module.

# Limitations
* Right now this prototype works only with jpeg document images (*.jpg).