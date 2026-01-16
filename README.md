This repo contains code to test,train and run the eyelid lesion classifier.

It is structured as follows:

First, you need to add the images to a new folder called Dataset, split into two folders labeled "benign" and "malignant".

Next, running histogram_match.py will create a new folder with the images histogram matched to the first image.

cross_validate_test.py performs k fold cross-validation, by default on the histogram matched data, although you can change this by modifiy the file names. This does not save any models, and since it trains a full CNN k times over it is recommend to use a GPU.

train_model.py trains the model on all data (by default the hist matched data) and saves the model to model/final_model.pth

It will override any other model unless you change the name in the script.

There is then two different GUIs for performing inference. Inference is fast, even on the largest model so no GPU is required. Histogram matching is not performed, which could be improved upon. The reason for this is that I did not want to store the first images histogram.
