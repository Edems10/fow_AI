import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time
import umap
from collections import Counter
from kmodes.kmodes import KModes
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from typing import List

from data_dragon import DataDragon
from lol_rules import CHAMPION_ROLES
from win_predictor.logistic_regression_win_predictor import LogisticRegressionWinPredictor
from win_predictor.random_forest_win_predictor import RandomForestWinPredictor
from win_predictor.gbt_win_predictor import GBTWinPredictor
from win_predictor.mlp_win_predictor import MLPWinPredictor
from win_predictor.win_feature_extractor import WinFeatureExtractor


def plot_k_modes(data: pd.DataFrame, game_ids: np.ndarray, data_dragon: DataDragon, output_folder: str) -> None:
    rows = []
    for id in game_ids:
        game_data = data.loc[data['id'] == id]
        rows.extend(game_data.iloc[:1].index)
    x = data.iloc[rows]

    champion_data = data_dragon.get_champion_data()
    champion_names = [champion for champion in sorted(champion_data['data'].keys())]

    team_columns = []
    for team in (1, 2):
        columns = []
        for champion_name in champion_names:
            column = np.array([champion_name in row.values
                               for idx, row
                               in x[[f't{team}_{role}_champion' for role in CHAMPION_ROLES]].iterrows()])
            columns.append(column)
        team_columns.append(np.stack(columns, axis=1))
    n_hot_x = np.concatenate(team_columns)
    reducer_2d = umap.UMAP(random_state=42)
    reducer_2d.fit(n_hot_x)

    reducer_3d = umap.UMAP(random_state=42, n_components=3)
    reducer_3d.fit(n_hot_x)

    champions = [x[[f't{team}_{role}_champion' for role in CHAMPION_ROLES]].values for team in (1, 2)]
    samples = np.concatenate(champions)
    X, Y = [], []
    for k in range(2, 30):
        km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=1).fit(samples)
        X.append(k)
        Y.append(km.cost_)
    plt.plot(X, Y, label='K-modes clustering')
    plt.xlabel('# of clusters')
    plt.ylabel('Cost')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'kmodes_cost.png'))
    plt.show()

    best_k = 4
    classes = KModes(n_clusters=best_k, init='Huang', n_init=5, verbose=1).fit_predict(samples)

    plt.scatter(reducer_2d.embedding_[:, 0], reducer_2d.embedding_[:, 1], c=classes, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(best_k + 1) - 0.5).set_ticks(np.arange(best_k))
    plt.title('UMAP projection of the clusters')
    plt.savefig(os.path.join(output_folder, 'umap_champion_clusters_2d.png'))
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(reducer_3d.embedding_[:, 0], reducer_3d.embedding_[:, 1], reducer_3d.embedding_[:, 2], c=classes, cmap='Spectral', s=5)
    plt.title('UMAP projection of the clusters')
    plt.savefig(os.path.join(output_folder, 'umap_champion_clusters_3d.png'))
    plt.show()


def measure_dataset_size_influence(df: pd.DataFrame,
                                   game_ids: List[str],
                                   data_dragon: DataDragon,
                                   models: List[dict],
                                   kf: KFold,
                                   output_folder: str) -> None:
    start = time.time()
    num_runs = 10
    samples_per_game = 1
    for model in models:
        accuracies = {}
        for data_fraction in (0.3, 0.5, 0.7, 1):
            print(data_fraction)
            train_accuracies = []
            test_accuracies = []
            for _ in range(num_runs):
                for split_number, (train_index, test_index) in enumerate(kf.split(game_ids)):
                    rows = []
                    for id in game_ids[train_index]:
                        if np.random.rand(1) > data_fraction:
                            continue
                        game_data = df.loc[df['id'] == id]
                        times = [timestep for timestep in game_data['time']]
                        chosen_samples = np.random.choice(times, samples_per_game, False)
                        for timestep in chosen_samples:
                            rows.extend(game_data.loc[game_data['time'] == timestep].index)

                    x_train = df.iloc[rows]
                    y_train = x_train.pop('winner').values
                    x_train.pop('id')

                    rows = []
                    for id in game_ids[test_index]:
                        game_data = df.loc[df['id'] == id]
                        rows.extend(game_data.index)
                    x_test = df.iloc[rows]
                    y_test = x_test.pop('winner').values
                    x_test.pop('id')

                    feature_extractor = WinFeatureExtractor(model['features'],
                                                            data_dragon,
                                                            True if 'normalize' in model and model['normalize'] else False)
                    model['model'] = LogisticRegressionWinPredictor(feature_extractor)
                    model['model'].train(x_train, y_train)

                    train_accuracy = accuracy_score(model['model'].predict(x_train), y_train)
                    print(model['name'], train_accuracy)
                    y_predicted = model['model'].predict(x_test)
                    test_accuracy = accuracy_score(y_predicted, y_test)
                    train_accuracies.append(train_accuracy)
                    test_accuracies.append(test_accuracy)

                    counts, correct = Counter(), Counter()
                    for i in range(len(x_test)):
                        timestep = int(x_test['time'].iloc[i])
                        counts[timestep] += 1
                        if y_test[i] == y_predicted[i]:
                            correct[timestep] += 1
            accuracies[data_fraction] = sum(train_accuracies) / kf.n_splits / num_runs, \
                                        sum(test_accuracies) / kf.n_splits / num_runs

        train_x, train_y, test_x, test_y = [], [], [], []
        for key, value in accuracies.items():
            train_x.append(key)
            test_x.append(key)
            train_y.append(value[0])
            test_y.append(value[1])
        # plt.plot(train_x, train_y, label=model['name'])
        plt.plot(test_x, test_y, label=model['name'])
    plt.xlabel("Fraction of the dataset available for training")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'dataset_size_influence.png'))
    plt.show()
    print(time.time() - start)


def measure_numsamples_influence(df: pd.DataFrame,
                                 game_ids: List[str],
                                 data_dragon: DataDragon,
                                 models: List[dict],
                                 kf: KFold,
                                 output_folder: str) -> None:
    start = time.time()
    num_runs = 10
    for model in models:
        accuracies = {}
        for samples_per_game in (1, 2, 4, 8, 16, 32, 64):
            print(samples_per_game)
            train_accuracies = []
            test_accuracies = []
            for _ in range(num_runs):
                for split_number, (train_index, test_index) in enumerate(kf.split(game_ids)):
                    rows = []
                    for id in game_ids[train_index]:
                        game_data = df.loc[df['id'] == id]
                        times = [timestep for timestep in game_data['time']]
                        chosen_samples = np.random.choice(times, samples_per_game, False)
                        for timestep in chosen_samples:
                            rows.extend(game_data.loc[game_data['time'] == timestep].index)

                    x_train = df.iloc[rows]
                    y_train = x_train.pop('winner').values
                    x_train.pop('id')

                    rows = []
                    for id in game_ids[test_index]:
                        game_data = df.loc[df['id'] == id]
                        rows.extend(game_data.index)
                    x_test = df.iloc[rows]
                    y_test = x_test.pop('winner').values
                    x_test.pop('id')

                    feature_extractor = WinFeatureExtractor(model['features'],
                                                            data_dragon,
                                                            True if 'normalize' in model and model['normalize'] else False)
                    model['model'] = LogisticRegressionWinPredictor(feature_extractor)
                    model['model'].train(x_train, y_train)

                    train_accuracy = accuracy_score(model['model'].predict(x_train), y_train)
                    print(model['name'], train_accuracy)
                    y_predicted = model['model'].predict(x_test)
                    test_accuracy = accuracy_score(y_predicted, y_test)
                    train_accuracies.append(train_accuracy)
                    test_accuracies.append(test_accuracy)

                    counts, correct = Counter(), Counter()
                    for i in range(len(x_test)):
                        timestep = int(x_test['time'].iloc[i])
                        counts[timestep] += 1
                        if y_test[i] == y_predicted[i]:
                            correct[timestep] += 1
            accuracies[samples_per_game] = sum(train_accuracies) / kf.n_splits / num_runs,\
                                           sum(test_accuracies) / kf.n_splits / num_runs

        train_x, train_y, test_x, test_y = [], [], [], []
        for key, value in accuracies.items():
            train_x.append(key)
            test_x.append(key)
            train_y.append(value[0])
            test_y.append(value[1])
        # plt.plot(train_x, train_y, label=model['name'])
        plt.plot(test_x, test_y, label=model['name'])
    plt.xlabel("Training samples per game")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'numsamples_influence.png'))
    plt.show()
    print(time.time() - start)


def main() -> None:
    data_folder = 'C:\\Users\\edems\\Documents\\Work\\fow_AI'
    data_dragon_folder = 'C:\\Users\\edems\\Documents\\Work\\fow_AI\\Game_data'
    #data_folder = sys.argv[1]
    dataset_folder = 'data_fow'
    output_folder = os.path.join('C:\\Users\\edems\\Documents\\Work\\fow_AI\\output', 'win_prediction')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # version = '11.15'
    version = '13.4'
    data_dragon = DataDragon(data_dragon_folder, version)

    df = pd.read_csv(os.path.join(data_folder,dataset_folder, 'win_dataset.csv'))
    kf = KFold(n_splits=5)
    game_ids = np.unique(df['id'])
    num_runs = 1
    samples_per_game = 1

    models = [{"name": "Baseline", "features": ['time', 'kills', 'turrets_total', 'monsters']},
              {"name": "Base", "features": ['kills', 'turrets_total', 'monsters']},
              # Single additions
              {"name": "Base - kills", "features": ['turrets_total', 'monsters']},
              {"name": "Base - turrets total", "features": ['kills', 'monsters']},
              {"name": "Base - monsters", "features": ['kills', 'turrets_total']},
              {"name": "Base + gold", "features": ['kills', 'turrets_total', 'monsters', 'gold']},
              {"name": "Base + level", "features": ['kills', 'turrets_total', 'monsters', 'level']},
              {"name": "Base + level mean", "features": ['kills', 'turrets_total', 'monsters', 'level_mean']},
              {"name": "Base + respawn", "features": ['kills', 'turrets_total', 'monsters', 'respawn']},
              {"name": "Base + alive", "features": ['kills', 'turrets_total', 'monsters', 'alive']},
              {"name": "Base + champions n-hot", "features": ['kills', 'turrets_total', 'monsters', 'champions_n_hot']},
              {"name": "Base + champion clusters", "features": ['kills', 'turrets_total', 'monsters', 'champions_k_modes']},
              {"name": "Base + epic buffs", "features": ['kills', 'turrets_total', 'monsters', 'epic_buffs']},
              {"name": "Base + inhibitors per lane", "features": ['kills', 'turrets_total', 'monsters', 'inhibitors_per_lane']},
              {"name": "Base + inhibitors total", "features": ['kills', 'turrets_total', 'monsters', 'inhibitors_total']},
              {"name": "Base + inhibitors respawn", "features": ['kills', 'turrets_total', 'monsters', 'inhibitors_respawn']},
              {"name": "Base + wards total", "features": ['kills', 'turrets_total', 'monsters', 'wards_total']},
              {"name": "Base + wards per type", "features": ['kills', 'turrets_total', 'monsters', 'wards_per_type']},
              # Modifications
              {"name": "Base - monsters + barons, dragons total", "features": ['kills', 'turrets_total', 'barons', 'dragons_total']},
              {"name": "Base - monsters + barons, dragons", "features": ['kills', 'turrets_total', 'barons', 'dragons']},
              {"name": "Base - monsters + epic buffs, barons, dragons total", "features": ['kills', 'turrets_total', 'epic_buffs', 'barons', 'dragons_total']},
              {"name": "Base - turrets total + turrets", "features": ['kills', 'turrets', 'monsters']},
              {"name": "Base - turrets total + turrets per lane", "features": ['kills', 'turrets_per_lane', 'monsters']},
              {"name": "Base - turrets total + turrets per tier", "features": ['kills', 'turrets_per_tier', 'monsters']},
              {"name": "Base - kills + gold, level mean", "features": ['turrets_total', 'monsters', 'gold', 'level_mean']},
              {"name": "Base - kills + gold, level mean, alive", "features": ['turrets_total', 'monsters', 'gold', 'level_mean', 'alive']},
              {"name": "Base - kills + gold, level mean, epic buffs", "features": ['turrets_total', 'monsters', 'gold', 'level_mean', 'epic_buffs']},
              # Combinations of modifications
              {"name": "Base + gold, level mean", "features": ['kills', 'turrets_total', 'monsters', 'gold', 'level_mean']},
              {"name": "Base + gold, epic buffs", "features": ['kills', 'turrets_total', 'monsters', 'gold', 'epic_buffs']},
              {"name": "Base + level mean, epic buffs", "features": ['kills', 'turrets_total', 'monsters', 'level_mean', 'epic_buffs']},
              {"name": "Base - monsters + gold, epic buffs, dragons total", "features": ['kills', 'turrets_total', 'gold', 'epic_buffs', 'dragons_total']},
              {"name": "Base - monsters + gold, barons, dragons total", "features": ['kills', 'turrets_total', 'gold', 'barons', 'dragons_total']},
              {"name": "All features", "features": ['kills', 'turrets', 'barons', 'dragons', 'epic_buffs', 'gold',
                                                         'respawn', 'level',  'inhibitors_per_lane', 'inhibitors_respawn', 'wards_per_type']}
              
              # FOW modifications 
              # TODO
              ]
    np.random.seed(42)
    start = time.time()
    split_metrics = {model['name']: {'correct': [], 'samples': []} for model in models}
    accuracies = {}
    for _ in range(num_runs):
        for split_number, (train_index, test_index) in enumerate(kf.split(game_ids)):
            rows = []
            for id in game_ids[train_index]:
                game_data = df.loc[df['id'] == id]
                times = [timestep for timestep in game_data['time']]
                # chosen_samples = np.random.choice(times, samples_per_game, False)
                chosen_samples = times
                for timestep in chosen_samples:
                    rows.extend(game_data.loc[game_data['time'] == timestep].index)
            x_train = df.iloc[rows]
            y_train = x_train.pop('winner').values
            x_train.pop('id')
            rows = []
            for id in game_ids[test_index]:
                game_data = df.loc[df['id'] == id]
                rows.extend(game_data.index)
            x_test = df.iloc[rows]
            y_test = x_test.pop('winner').values
            x_test.pop('id')
            for model in models:
                feature_extractor = WinFeatureExtractor(model['features'],
                                                        data_dragon,
                                                        True if 'normalize' in model and model['normalize'] else False)
                model['model'] = LogisticRegressionWinPredictor(feature_extractor)
                model['model'].train(x_train, y_train)

                train_accuracy = accuracy_score(model['model'].predict(x_train), y_train)
                print(model['name'], train_accuracy)
                y_predicted = model['model'].predict(x_test)
                test_accuracy = accuracy_score(y_predicted, y_test)
                if 'train_accuracies' not in model:
                    model['train_accuracies'] = []
                    model['test_accuracies'] = []
                model['train_accuracies'].append(train_accuracy)
                model['test_accuracies'].append(test_accuracy)

                counts, correct = Counter(), Counter()
                for i in range(len(x_test)):
                    timestep = int(x_test['time'].iloc[i])
                    counts[timestep] += 1
                    if y_test[i] == y_predicted[i]:
                        correct[timestep] += 1
                split_metrics[model['name']]['samples'].append(counts)
                split_metrics[model['name']]['correct'].append(correct)

    plt.figure(figsize=(9, 6))
    print(time.time() - start)
    with open(os.path.join(output_folder, 'model_accuracies.csv'), 'w') as f:
        columns = ['Features', 'Train accuracy', 'Test accuracy']
        print('\t'.join(columns), file=f)
        for model in models:
            metrics = split_metrics[model['name']]
            all_game_times = sorted([key for samples in metrics['samples'] for key in samples.keys()])
            x_plot, means, stds = np.zeros((3, len(all_game_times)))
            for i, game_time in enumerate(sorted(all_game_times)):
                x_plot[i] = (game_time / 1000.0 / 60.0)
                means[i] = (sum(correct[game_time] for correct in metrics['correct']) / sum(count[game_time] for count in metrics['samples']))
                fold_accuracies = [(correct[game_time] / count[game_time] - means[i]) ** 2 for count, correct in zip(metrics['samples'], metrics['correct']) if game_time in count]
                stds[i] = np.sqrt(sum(fold_accuracies) / len(fold_accuracies))
            means = np.array(means)
            stds = np.array(stds)
            plt.plot(x_plot, means, label=model['name'])
            train_accuracy = sum(model['train_accuracies']) / kf.n_splits / num_runs
            test_accuracy = sum(model['test_accuracies']) / kf.n_splits / num_runs
            print('\t'.join([model['name'], str(train_accuracy), str(test_accuracy)]), file=f)
            plt.fill_between(x_plot, means - stds, means + stds, alpha=.1)
    plt.ylabel("Prediction accuracy")
    plt.xlabel("Game time (minutes)")
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'prediction_accuracy_over_time.png'))
    plt.show()


if __name__ == '__main__':
    main()
