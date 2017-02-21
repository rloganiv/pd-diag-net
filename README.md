pd-diag-net
===========

Authors: Anthony Chen, Shreya Chippagiri, Robert Logan, Pratik Shetty

pd-diag-net is a deep neural classifier which uses vocal samples to predict
whether a patient has Parkinsons Disease. The model is trained using the
[Parkinsons Disease Handwriting Database](http://bdalab.utko.feec.vutbr.cz/)
collected by the BDALab Research Group at Brno University of Technology.


Installation
------------

### System Setup
Users will need to have python2.7 installed.

### Python Setup
The model is built using the Keras library in python; all of the dependencies
are in the `requirements.txt` file. You can install these in a virtual
environment by running:
```
python -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
When you are finished working with the model you can run:
```
deactivate
```
to disable the virtual environment.


Usage
-----

### Step 1: Download and preprocess the data
In order to obtain access to the PaHaW database, you will need to fill out a
licensing agreement. For more details please see the downloads section on
[this website](http://bdalab.utko.feec.vutbr.cz/).

Once the dataset has been downloaded, extract the compressed dataset into the
project folder - i.e. 'PaHaW/' should be a directory at the root level.

The dataset can then be loaded into python by adding:
```
import process
dataset = process.load_dataset()
```
to your script.

### Step 2: Training and evaluating the model
As of now, evaluation of our model is done using k-fold cross validation. As such, training
are tightly coupled. 

K-Fold accuracy can be done by doing the following:
'''
import model
model.evaluate_model()
'''


### Step 3: Run the model
TBD - Something about reproducing the accuracy metrics included in our paper as
well as instructions for running the model on new data.

