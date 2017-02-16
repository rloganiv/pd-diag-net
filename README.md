pd-diag-net
===========

Authors: Anthony Chen, Shreya Chippagiri, Robert Logan, Pratik Shetty

pd-diag-net is a deep neural classifier which uses vocal samples to predict
whether a patient has Parkinsons Disease. The model is trained using the
[Parkinsons Speech Dataset](https://archive.ics.uci.edu/ml/datasets/Parkinson+Speech+Dataset+with++Multiple+Types+of+Sound+Recordings)
publicly available for download from the UCI Machine Learning Repository.


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
TBD

### Step 2: Train the model
TBD

### Step 3: Run the model
TBD - Something about reproducing the accuracy metrics included in our paper as
well as instructions for running the model on new data.

