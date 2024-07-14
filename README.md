## Description

This repository holds the experimental part of my coursework on topic 'Investigation of Transfer Learning for Multiclass Satellite Image Classification'. The main idea of the research was to apply transfer learning algorithm using two different satellite imagery datasets and assess metrics of trained model using plain, normalized and cdf-normalized feature map.

## Prerequisities
Due to github limitations on size of data to upload, before executing code found in this repository, download of Sentinel-4 and Dubai segmentation dataset is required:
1. In repository, crete `data/sat4` folder and upload `sat-4-full.mat` file including first dataset information to it. The file can be downloaded from [kaggle](https://www.kaggle.com/datasets/crawford/deepsat-sat4)
2. In `data/SemanticSegmentationDataset` folder save tiles downloaded from [Dubai dataset](https://humansintheloop.org/resources/datasets/semantic-segmentation-dataset-2/) and generate a processed dataset using `DubaiDataGenerator` class passing it list of all images in tiles along with list of their masks.

## Flow of the program
1. Upload of required datasets.
2. Utilization of resnet18 for training on SAT-4 dataset, saving the pretrained model.
3. Running pretrained model with Dubai dataset, extracting features.
4. Using extracted features as input to train simple 2 layer model.
5. Evaluating results.