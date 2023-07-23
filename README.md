# Fantasy AI

## Data dragon

Download: https://ddragon.leagueoflegends.com/cdn/dragontail-11.15.1.tgz

Available versions: https://ddragon.leagueoflegends.com/api/versions.json

Replace the url at https://ddragon.leagueoflegends.com/cdn/dragontail-9.3.1.tgz

## Create datasets

1. Move batches of raw data and the extracted data dragon folder to a 
`DATA_FOLDER` of your choice.
2. Run `aggregate.py DATA_FOLDER` to get aggregated discrete time steps from the raw events.
The script also adds extra features which can be mined from but are not directly
in the raw data.
### Win prediction dataset
3. Run `create_win_dataset.py DATA_FOLDER` to create a dataset for win prediction. This
extracts features from the aggregated dataset for win prediction and saves them
to a csv file. It creates a `data/win_dataset.csv` file.
### Macro prediction dataset
3. Run `create_macro_dataset.py DATA_FOLDER` to create a dataset for macro prediction. It adds
targets and spatial features to the aggregated game states and transforms the
features so that they can easily be used with a neural network.
4. Run `split_macro_dataset.py DATA_FOLDER` which text files `train.txt`, `valid.txt` and
`test.txt` with ids of games for each split.
5. Run `macro_dataset_to_samples.py DATA_FOLDER` which stores all the sequences of the given
history size on the disk so that they can be loaded with a random access. The
samples are split into train, test and valid folders as per the output of the
previous step.

## Train models
### Win prediction

1. Run `predict_win.py DATA_FOLDER` which tests different models and saves their results in
`output/win_prediction/model_accuracies.csv`.
### Macro prediction
1. Run `predict_macro.py DATA_FOLDER`.