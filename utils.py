import json
import numpy as np
import os
from typing import Any, Generator, List, Optional
from multiprocessing.pool import ThreadPool


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
        raise NotImplementedError
    


def get_game_version_map(data_folder: str) -> dict:
    headers_folder = os.path.join(data_folder, 'Headers_Games')
    
    missing_folder = os.path.join(data_folder, 'Batch')
    #missing_folder = os.path.join(data_folder, 'Missing_Versions')
    
    if not os.path.isdir(missing_folder):
        os.mkdir(missing_folder)

    id_versions = {}
    for header_file in os.listdir(headers_folder):
        game_id = header_file.split('.')[0].split('_')[1]
        success, version = get_version(os.path.join(headers_folder, header_file))
        if success:
            id_versions[game_id] = version
        else:
            for i in range(1,6):
                game_iter = f'_{str(i)}.json'
                success, version = get_version(os.path.join(missing_folder, header_file.replace('.json', game_iter)))
                if success:
                    id_versions[game_id] = version
                    break
            else:
                id_versions[game_id] = "MISSING_HEADER_FILE"

    return id_versions


def get_file_events(fn: str) -> List[dict]:
    with open(fn, 'r') as f:
        return json.load(f)['events']


def read_matches_from_batches(data_folder: str, version: str, batch_size: int = 400, threads: int = 8) -> Generator[List[dict], Any, None]:
    sorting_key = lambda e: int(e['sequenceIndex'])

    version_map = get_game_version_map(data_folder)

    # Load matches in parallel using multiple threads
    with ThreadPool(threads) as pool:
        for dir in os.listdir(data_folder):
            if dir.startswith('Batch') and not dir.endswith('.zip'):
                batch_folder = os.path.join(data_folder, dir)
                game_files = [os.path.join(batch_folder, file) for file in os.listdir(batch_folder) if file.endswith('.json')]
                events_buffer = []
                for i in range(0, len(game_files), batch_size):
                    end_idx = min(i + batch_size, len(game_files))
                    events_buffer.extend(zip(game_files[i:end_idx], pool.map(get_file_events, game_files[i: end_idx], chunksize=20)))
                    current_start = 0
                    previous_id = None
                    for j in range(len(events_buffer)):
                        _, tail = os.path.split(events_buffer[j][0])
                        id = tail.split('_')[1]
                        if previous_id is not None and id != previous_id:
                            match_events = sorted([event for events in events_buffer[current_start:j] for event in events[1]], key=sorting_key)
                            if match_events:
                                if len(set(event['sequenceIndex'] for event in match_events)) != len(match_events):
                                    print("Found a broken match with different events sharing sequenceIndex.")
                                else:
                                    try:
                                        if version_map[id] == version:
                                            yield match_events
                                        else:
                                            print(f"{id} has unwanted version {version_map[id]}")
                                    except Exception as e:
                                        print(f"1-issue found with: {e}")
                            else:
                                print(f"Found a broken match with no events: {previous_id}")
                            current_start = j
                        previous_id = id
                    events_buffer = events_buffer[current_start:]
                if events_buffer:  # yield the remaining game
                    _, tail = os.path.split(events_buffer[0][0])
                    id = tail.split('_')[1]
                    match_events = sorted([event for events in events_buffer for event in events[1]], key=sorting_key)
                    if len(set(event['sequenceIndex'] for event in match_events)) != len(match_events):
                        print("Found a broken match with different events sharing sequenceIndex.")
                    else:
                        try:
                            if version_map[id] == version:
                                yield match_events
                            else:
                                print(f"{id} has unwanted version {version_map[id]}")
                        except Exception as e:
                            print(f"2-issue found with: {e}")


def load_game_data(data_folder: str) -> Generator[dict, Any, None]:
    for game_fn in os.listdir(data_folder):
        with open(os.path.join(data_folder, game_fn)) as f:
            yield game_fn.split('.')[0], json.load(f)


def determine_champion_roles(game: dict) -> dict:
    """
    Implements the champion role categorization pseudocode from
    Smart kills and worthless deaths: eSports analytics for League of Legends.
    """
    mapping = {}
    for team in (1, 2):
        minions_killed_lvl1 = np.zeros(5)
        minions_killed_lvl6 = np.zeros(5)
        neutral_minions_killed_lvl1 = np.zeros(5)
        neutral_minions_killed_lvl6 = np.zeros(5)
        near_each_other = np.zeros((5, 5))
        near_mid = np.zeros(5)  # FIXME: As long as minion data is not available, we have to use this
        format = game['format']
        minions_killed_index = format['state']['champion_stats'].index('minions_killed')
        neutral_minions_killed_index = format['state']['champion_stats'].index('neutral_minions_killed')
        position_index = format['state']['champion_state'].index('position')
        level_index = format['state']['champion_state'].index('level')
        for timestep_data in game['data']:
            timestep_state = timestep_data['state']
            if 'champion_state' not in timestep_state:
                continue
            if timestep_data['gameTime'] <= 10 * 60 * 1000:
                positions = [timestep_state['champion_state'][i + (team - 1) * 5][position_index] for i in range(5)]
                for c1 in range(5):
                    for c2 in range(5):
                        if c1 != c2 and np.linalg.norm(np.array([positions[c1][coord] - positions[c2][coord] for coord in ('x', 'z')])) < 0.1 * 15000:
                            near_each_other[c1, c2] += 2
                for i in range(5):
                    mid_distance = np.linalg.norm(np.array([7500 - positions[i]['x'], 7500 - positions[i]['z']]))
                    if mid_distance < 0.1 * 15000:
                        near_mid[i] += 1
            levels = np.array([timestep_state['champion_state'][i + (team - 1) * 5][level_index] for i in range(5)])
            level1 = levels == 1
            level6 = levels <= 6
            if level6.any():
                minions_killed = np.array([timestep_state['champion_stats'][i + (team - 1) * 5][minions_killed_index] for i in range(5)])
                neutral_minions_killed = np.array([timestep_state['champion_stats'][i + (team - 1) * 5][neutral_minions_killed_index] for i in range(5)])
                minions_killed_lvl1[level1] = minions_killed[level1]
                minions_killed_lvl6[level6] = minions_killed[level6]
                neutral_minions_killed_lvl1[level1] = neutral_minions_killed[level1]
                neutral_minions_killed_lvl6[level6] = neutral_minions_killed[level6]
            elif timestep_data['gameTime'] >= 10 * 60 * 1000:
                break
        neutral_minions_killed = neutral_minions_killed_lvl6 - neutral_minions_killed_lvl1
        assert (neutral_minions_killed == neutral_minions_killed.max()).sum() == 1
        jungle = np.argmax(neutral_minions_killed)

        minions_killed_lvl6[jungle] = np.inf  # Jungler cannot also be support
        minions_killed = minions_killed_lvl6 - minions_killed_lvl1
        assert (minions_killed == minions_killed.min()).sum() == 1
        support = np.argmin(minions_killed)

        near_mid[[jungle, support]] = 0  # Support and jungler cannot also be mid
        assert (near_mid == near_mid.max()).sum() == 1
        mid = np.argmax(near_mid)

        near_each_other[support, [jungle, mid]] = 0
        assert (near_each_other[support] == near_each_other[support].max()).sum() == 1
        adc = np.argmax(near_each_other[support])

        top = [i for i in range(5) if i not in [jungle, support, mid, adc]][0]
        mapping[team] = {'jungle': jungle + 5 * (team-1),
                         'support': support + 5 * (team-1),
                         'mid': mid + 5 * (team-1),
                         'adc': adc + 5 * (team-1),
                         'top': top + 5 * (team-1)}
        # if not (jungle == 1 and support == 4 and mid == 2 and adc == 3 and top == 0):
        #     print(mapping[team])
    return mapping
