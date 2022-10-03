# img_segAndcorr
img's segmentation and correlation

The directory dataset4segmentation include the dataset for semantic segmentation to binarize pixels,
and the data4correction contains images that need to be corrected and the results after corrected.

best_model_174.pth is the trained semantic segmentation model, which can be loaded and used directly in test.py
If you need to retrain the semantic segmentation model, you can use the data in dataset4segmentation to train in train.py

the drift_correction.cpp is a cpp file,which calculates the drift distance of the binarized image and corrects it.This file
need run in environment of cpp.

How does the program run?
firstly,
