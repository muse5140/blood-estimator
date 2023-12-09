# blood-estimator
  To estimate blood amount from used gauzes.

**This repo structure**:
```
root
├── data
│   ├── data_raw: raw data is stored here. File names must be appended by date.
│   ├── data_processed: processed data is stored here. File names must be appended by date.
│   └── data_dataset:train/test/validation set is stored here. Data here will be used for models training. File names must be appended by date.
├── models: store ML models.
└── src
    ├── process_data: script to process data as well as performing data split.
    ├── model_training: script/notebook to train ML model and store the model in models/
    └── analyse_result: script/notebook for analysing experiment result as well as visualization.
```
Note: Since this repo is public. Data won't be uploaded into github.
