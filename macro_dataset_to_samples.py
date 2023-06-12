import json
import os
import sys
from tqdm import tqdm
from typing import List


def save_sample(data: List[dict], filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(data, f)


def split_file_to_samples(filename: str, dataset_folder: str, output_dataset_folder: str, history_length: int) -> None:
    id = filename.split('.')[0]
    with open(os.path.join(dataset_folder, filename), 'r') as game_file:
        history = []
        i = 1
        for line in game_file:
            history.append(json.loads(line))
            if len(history) > history_length:
                history = history[1:]
            if len(history) == history_length:
                with open(os.path.join(output_dataset_folder, f'{id}_{i}.jsonl'), 'w') as out_file:
                    for history_item in history:
                        out_file.write(json.dumps(history_item) + '\n')
                i += 1


def main() -> None:
    #data_folder = sys.argv[1]
    data_folder = os.path.join(os.path.dirname(__file__),'Game_data')
    dataset_folder = os.path.join(data_folder, 'macro_dataset')

    history_length = 4
    output_folder = dataset_folder + f'_{history_length}'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    dataset_names = ('train', 'test', 'valid')
    for dataset_name in dataset_names:
        output_dataset_folder = os.path.join(output_folder, dataset_name)
        if not os.path.exists(output_dataset_folder):
            os.mkdir(output_dataset_folder)
        with open(os.path.join(data_folder, 'macro_prediction_split', f'{dataset_name}.txt'), 'r') as f:
            for line in tqdm(f):
                split_file_to_samples(line.strip(), dataset_folder, output_dataset_folder, history_length)


if __name__ == '__main__':
    main()
