# SealNet

Python 3.6 is what I use and unsure about this projects compatibility with other versions.

To setup run:
`pip install .`

#### Project structure outline of important things
    .
    ├── data                   # label metadata
    │   ├── TrainingAnimals_WithSightings_updating.csv          # all CHESS hotspots with updated boxes
    │   └── updated_seals.csv                                   # only seals that have not been marked as 'removed' and have updated labels
    ├── dataset                # generate datasets/chips for training
    │   └── label_formats      # for generating various label formats
    ├── metrics                # evaluation tools
    │   └── ensemble.py        # ensemble detections from multiple models run through VIAME
    ├── models                 # various models for training/testing
    ├── scripts                # random scripts
    ├── setup.py               # project path/dependency setup stuff
    └── README.md

#### Data flow `dataset/*`
Goal was to create a way to go from a ground truth csv to a dataset of chips that could be trained.  Since we chip we need to also be able to place chips back into their original image.

* `dataset/dataset_config.py`: defines transformations for a dataset, data location, chip size, name, and other things.
* `dataset/generate_data_outline.py`: generates the output dataset directory and places the config file in that directory along with a csv of test susbset from original data and train subset.
* `dataset/generate_data_items.py`: given a directory created by the `generate_data_outline.py` takes the test.csv and train.csv and generates the training chips using the size given in the config
* `dataset/generate_background_chips.py`: generates a given number of test and train background chips
* `dataset/label_formats/*`: generate various label formats for the generated dataset

#### Ensemble method
Idea here is to improve label quality and results in general(reduce FP rate) by ensembling multiple models.  Currently the goal is to use this to improve the ground truth dataset by finding missing labels.

**Implementation:** `metrics/ensemble.py`
* Do NMS each image for each model (do not pool all model detections for one image then nms)
* Create image mask with n channels where n is the number of models we are ensembling
* Fill in pixels on each channel with corresponding confidence^2 of that pixel and class id
* If one pixel has multiple detections use one with highest confidence
* We then take the n-channel mask and turn it into a 1-channel binarized mask
   * Sum across channels for each pixel to get flat mask
   * Take 90th percentile of non-zero values as a threshold
   * Set everything above thresh to 1 and below to 0
* Extract bounding boxes from the binary mask
* Find correlating class id for each box 
* Set new confidence to the sqrt of the mean confidences at that pixel # TODO sorta but not totally l2 norm, should it be?

#### Models `models/*`
For different models, I think all my tensorflow stuff is currently local but if you made tensorflow models you would put them here and use `dataset/label_formats/tf_records.py` to create the tensorflow labels.

Darknet is the main model I've been using and linked in models is my fork readicculus/darknet.  I've added python tools to do multi threading inference which was used for ensembling mutliple models trained in darknet. 



This current project still in working stages but main idea is to be able to easily keep track of data, models, and results which
I have been struggling with in the previous repo.  The more specific short-term goal is label refinement and providing tools for 
labeling new data with as little manual work as possible.  Also polar bears.

Old project: https://github.com/readicculus/Bears-n-Seals (view at own risk)