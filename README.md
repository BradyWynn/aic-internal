this is the first place submission for the ai community internal kaggle competition
https://www.kaggle.com/competitions/ai-community-internal-comp

the image dataset is too large to include in the repository so to reproduce the results you need to download the dataset from the competiton 
make sure to have the test and train folders in the directory and delete the train.csv file inside the train folder

there are two seperate models that need to be trained, the unet for segmentation and cnn for classification
please note that i did not write the unet class myself and it was taken from https://github.com/jaxony/unet-pytorch/blob/master/model.py

on my rtx 3070 ti the unet took 38 minutes to train and the cnn took 11.5 seconds, it may take longer if you have a weaker gpu or train on cpu

the test.ipynb is what generates the submission file and needs both the unet and cnn to be trained and inside the /models folder
