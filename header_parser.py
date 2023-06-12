import datetime as dt
import json
import matplotlib.dates
import os
from collections import Counter
from matplotlib import pyplot as plt
from typing import Optional

from utils import load_game_data


def plot_version_games_count(game_versions: Counter) -> None:
    x, y = [], []
    ordered_items = sorted(game_versions.items(), key=lambda i: int(i[0][:2]) * 100 + int(i[0][3:]))
    for key, value in ordered_items:
        x.append(key)
        y.append(value)
    plt.figure(figsize=(14, 8))
    plt.bar(x, y)
    plt.title("Number of games per game version")
    plt.savefig("output/version_distribution.png", facecolor='white', transparent=False)
    plt.show()


def plot_version_over_time(patch_distribution: dict) -> None:
    x, y = [], []
    for time, version in patch_distribution.items():
        x.append(matplotlib.dates.date2num(time))
        y.append(int(version[:2]) * 100 + int(version[3:]))
    plt.figure(figsize=(16, 8))
    plt.plot_date(x, y, markersize=1)
    plt.title("Game versions over time")
    plt.savefig("output/versions_over_time.png", facecolor='white', transparent=False)
    plt.show()


def plot_faulty_games_over_time(fault_distribution: [dt.datetime]) -> None:
    x = [matplotlib.dates.date2num(date) for date in fault_distribution]
    y = [1 for _ in x]
    plt.figure(figsize=(16, 8))
    plt.plot_date(x, y, markersize=1)
    plt.title("Distribution of games without game_info event over time")
    plt.savefig("output/faulty_games_over_time.png", facecolor='white', transparent=False)
    plt.show()


def find_pick_phase_differences(d1: dict, d2: dict) -> None:
    for key, value in d1.items():
        if key not in d2:
            print(f"{key} not in d2")
        elif type(value) is dict:
            find_pick_phase_differences(d1[key], d2[key])
        elif type(value) is list:
            if len(value) != len(d2[key]):
                print(f"{key} different list lengths!")
            else:
                for v1, v2 in zip(value, d2[key]):
                    find_pick_phase_differences(v1, v2)
        elif value != d2[key]:
            print(f"Found a difference! {key}: {value}!={d2[key]}")


def get_version(file: str) -> [bool, Optional[str]]:
    with open(file, 'r') as f:
        headers = json.load(f)
    game_info_events = 0
    for event in headers['events']:
        if event['rfc461Schema'] == 'game_info':
            game_info_events += 1
            game_version = event['gameVersion']
            major, minor, *_ = game_version.split('.')
            version = major + '.' + minor
            return True, version

    if game_info_events == 0:
        return False, None
    elif game_info_events > 1:
        print(file, game_info_events)


def main() -> None:
    headers_folder = 'D:/fantasyai/data/Headers_Games'
    missing_folder = 'D:/fantasyai/data/Missing_Versions'
    game_versions = Counter()
    total_games, faulty_games = 0, 0
    faulty_game_names = []
    fault_datetimes = []
    version_distribution = {}
    id_versions = {}
    for header_file in os.listdir(headers_folder):
        game_id = header_file.split('.')[0].split('_')[1]
        total_games += 1
        success, version = get_version(os.path.join(headers_folder, header_file))
        if success:
            game_versions[version] += 1
            id_versions[game_id] = version
        else:
            success, version = get_version(os.path.join(missing_folder, header_file.replace('.json', '_1.json')))
            if success:
                game_versions[version] += 1
                id_versions[game_id] = version
            else:
                faulty_games += 1

    # Check whether the games we have are of the correct version
    for id, game in load_game_data('D:/fantasyai/data/aggregated'):
        game_version = id_versions[id]
        if game_version != '11.15':
            print(id, game_version)

    with open(f"output/games_without_game_info_event.csv", 'w') as f:
        for fn in faulty_game_names:
            print(fn, file=f, end=',\n')
    print(f"Total games: {total_games} Games without game_info event: {faulty_games}")
    # print([f"{key // 100}.{key % 100}: {item}"for key, item in game_versions.items()])
    print(game_versions)
    plot_version_games_count(game_versions)
    plot_version_over_time(version_distribution)
    plot_faulty_games_over_time(fault_datetimes)


if __name__ == '__main__':
    main()
