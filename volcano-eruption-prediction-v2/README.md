# Predicting Eruptive Events at Volcanoes from Earthquake Data

## Project objective
Build a machine-learning warning system using earthquake characteristics
from the 2018 Kilauea earthquake sequence.

## Dataset
The supplied catalog contains earthquake date/time, latitude, longitude,
depth, magnitude and event ID.

## Project-defined prediction task
The supplied catalog does not contain an explicit eruption 0/1 label.
Therefore, this project defines a forward-looking warning target:

- **1 (Within 24 hours):** the earthquake occurred within 24 hours before
  the defined eruption onset.
- **0 (More than 24 hours):** the earthquake occurred more than 24 hours
  before the defined eruption onset.

The project uses **2018-05-03 00:00:00** as the eruption-onset reference
for the 2018 Kilauea sequence.

## Features
- Latitude
- Longitude
- Depth
- Magnitude

## Machine learning model
Random Forest Classifier.

## Evaluation
- Accuracy
- ROC-AUC
- Confusion Matrix
- Precision, Recall and F1-score
- Feature Importance

## Run the project

```bash
pip install -r requirements.txt
python volcano_prediction.py
```

## Important limitation
The earthquake catalog itself does not provide the eruption label.
The eruption onset and 24-hour warning target are project-defined from
the known eruption sequence. This limitation must be clearly stated in
the final report and presentation.

## Reference
The project is inspired by the supplied research paper, which studies
classification of eruptive state and forecasting time to eruption using
earthquake data and machine-learning methods.
