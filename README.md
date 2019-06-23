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
    ├── models                 # various models for training/testing
    ├── scripts                # random scripts
    ├── setup.py               # project path/dependency setup stuff
    └── README.md


Old project: https://github.com/readicculus/Bears-n-Seals