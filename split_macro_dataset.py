import os
import sys
from sklearn.model_selection import train_test_split


def main() -> None:
    data_folder = os.path.join(os.path.dirname(__file__),'Game_data')
    output_folder = os.path.join(os.path.dirname(__file__),'Game_data', 'macro_prediction_split_fow')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    dataset_folder = os.path.join(data_folder, 'macro_fow')
    games = os.listdir(dataset_folder)
    train_files, test_files = train_test_split(games, test_size=0.2, train_size=0.8)
    valid_files, test_files = train_test_split(test_files, test_size=0.5, train_size=0.5)
    for dataset_name, filenames in (('train', train_files),
                                    ('valid', valid_files),
                                    ('test', test_files)):
        with open(os.path.join(output_folder, f'{dataset_name}.txt'), 'w') as f:
            for filename in filenames:
                print(filename, file=f)


if __name__ == '__main__':
    main()
