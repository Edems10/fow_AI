import json
import sys

import numpy as np
import os
from typing import List
from tqdm import tqdm

from data_dragon import DataDragon
from utils import load_game_data
from lol_rules import EXTRA_SUMMONER_SPELL_NAMES, INHIBITOR_RESPAWN_TIME, MAP_SIZE, WARD_TYPES
from get_building_positions import BUILDING_POSITONS


def add_targets(game: dict, position_prediction_split_n: int, discount_factor: float, time_horizon: int) -> None:
    num_targets = len(game['data']) - 1  # The last timestep cannot be used for training
    map_sector_edge = MAP_SIZE / position_prediction_split_n
    targets = {'position': np.zeros((num_targets, 10)),
               'minions': np.zeros((num_targets, 10)),
               'monsters': np.zeros((num_targets, 10)),
               'kills': np.zeros((num_targets, 10)),
               'turrets': np.zeros((num_targets, 10)),
               'inhibitors': np.zeros((num_targets, 10)),
               'heralds': np.zeros((num_targets, 10)),
               'dragons': np.zeros((num_targets, 10)),
               'barons': np.zeros((num_targets, 10))}

    # dragon_type = None
    cs = np.zeros((10, 2))  # Minions and neutral monsters killed per champion in the current state
    format = game['format']
    for timestep, timestep_data in enumerate(game['data']):
        timestep_state = timestep_data['state']
        time_horizon_start = max(timestep - time_horizon, 0)
        # Used to increment previous states whan an event appears
        discount_factor_scaling = discount_factor ** np.arange(min(time_horizon, timestep))
        for champion_index in range(10):
            champion_state = timestep_state['champion_state'][champion_index]
            champion_stats = timestep_state['champion_stats'][champion_index]
            # Handle position target
            position = champion_state[format['state']['champion_state'].index('position')]
            cell_index = min((position['x'] // map_sector_edge), position_prediction_split_n - 1) +\
                         (min(position['z'] // map_sector_edge, position_prediction_split_n - 1) * position_prediction_split_n)
            targets['position'][timestep - 1, champion_index] = cell_index
            # Handle minion and monster targets
            minions_killed = champion_stats[format['state']['champion_stats'].index("minions_killed")]
            monsters_killed = champion_stats[format['state']['champion_stats'].index("neutral_minions_killed")]
            minions_diff = minions_killed - cs[champion_index, 0]
            monsters_diff = monsters_killed - cs[champion_index, 1]
            targets['minions'][time_horizon_start: timestep, champion_index] += discount_factor_scaling * minions_diff
            targets['monsters'][time_horizon_start: timestep, champion_index] += discount_factor_scaling * monsters_diff
            cs[champion_index] = minions_killed, monsters_killed

        for event_type, event in timestep_data['events']:
            if event_type == 'building_destroyed':
                participants = set([event[format['events'][event_type].index('lastHitter')]] + \
                                   event[format['events'][event_type].index('assistants')])
                participants = [participant - 1 for participant in participants if participant != 0]
                building_type = event[format['events'][event_type].index('buildingType')]
                if 'turret' in building_type:
                    targets['turrets'][time_horizon_start: timestep, participants] += discount_factor_scaling[:, np.newaxis]
                elif building_type == 'inhibitor':
                    targets['inhibitors'][time_horizon_start: timestep, participants] += discount_factor_scaling[:, np.newaxis]
            elif event_type == 'champion_kill':
                participants = set([event[format['events'][event_type].index('killer')]] + \
                                   event[format['events'][event_type].index('assistants')])
                participants = [participant - 1 for participant in participants if participant != 0]
                targets['kills'][time_horizon_start: timestep, participants] += discount_factor_scaling[:, np.newaxis]
            elif event_type == 'epic_monster_kill':
                participants = set([event[format['events'][event_type].index('killer')]] +\
                               event[format['events'][event_type].index('assistants')])
                participants = [participant - 1 for participant in participants if participant != 0]
                monster_type = event[format['events'][event_type].index('monsterType')]
                if monster_type == 'dragon':
                    targets['dragons'][time_horizon_start: timestep, participants] += discount_factor_scaling[:, np.newaxis]
                elif monster_type == 'baron':
                    targets['barons'][time_horizon_start: timestep, participants] += discount_factor_scaling[:, np.newaxis]
                elif monster_type == 'riftHerald':
                    targets['heralds'][time_horizon_start: timestep, participants] += discount_factor_scaling[:, np.newaxis]
    for timestep in range(num_targets):
        game['data'][timestep]['targets'] = {key: value[timestep].tolist() for key, value in targets.items()}
    game['data'].pop()  # Remove the last timestep for which we have no target


def add_features(game: dict, map_split_n: int) -> None:
    format = game['format']
    map_sector_edge = MAP_SIZE / map_split_n

    format['state']['map'] = {'global': ['spatial_features']  # array[map_split_n, map_split_n, 12]
                              }
    # How many champions, turrets, inhibitors and wards of each type are present in each sector for each team
    map_feature = np.zeros((12, map_split_n, map_split_n))

    for building_type, positions in BUILDING_POSITONS.items():
        index = 1 if 'turret' in building_type else 2  # Turret or inhibitor
        team = int(building_type.split('_')[-1])
        index += 6 * team
        for position in positions:
            sector = np.array(position) // map_sector_edge
            map_feature[(index,) + tuple(sector.astype(int))] += 1
    respawns = []
    for timestep_data in game['data']:
        timestep_state = timestep_data['state']
        game_time = timestep_data['gameTime']
        if 'champion_state' not in timestep_state:
            continue
        for sector, index, respawn_time in respawns:
            if respawn_time <= game_time:
                map_feature[(index,) + tuple(sector.astype(int))] += 1
        respawns = [respawn for respawn in respawns if respawn[-1] > game_time]
        for event_type, event in timestep_data['events']:
            if event_type == 'building_destroyed':
                building_type = event[format['events'][event_type].index('buildingType')]
                team_id = event[format['events'][event_type].index('teamID')] // 100 - 1
                position = event[format['events'][event_type].index('position')]
                index = 1 if 'turret' in building_type else 2
                index += 6 * team_id
                sector = np.array((position['x'], position['z'])) // map_sector_edge
                map_feature[(index,) + tuple(sector.astype(int))] -= 1
                if 'inhibitor' in building_type:
                    respawns.append([sector, index, game_time + INHIBITOR_RESPAWN_TIME])
        # Clear champion and ward positions
        map_feature[[0, 3, 4, 5, 6, 9, 10, 11], :, :] = 0
        for team in range(2):
            champion_feature_index = team * 6
            ward_feature_indices = {ward_type: 3 + i + team * 6
                                  for i, ward_type in enumerate(WARD_TYPES)}
            for champion_index in range(5 * team, 5 * (team + 1)):
                position = timestep_state['champion_state'][champion_index][format['state']['champion_state'].index('position')]
                champ_sector = np.array((position['x'], position['z'])) // map_sector_edge
                map_feature[(champion_feature_index,) + tuple(champ_sector.astype(int))] += 1
                for ward_type in WARD_TYPES:
                    ward_feature = f"{ward_type}_wards"
                    for ward in timestep_state[ward_feature][champion_index]:
                        x, y = ward[format['state'][ward_feature].index('x')], ward[format['state'][ward_feature].index('y')]
                        if x == 0 and y == 0:  # Zero padded
                            continue
                        ward_sector = np.array((x, y)) // map_sector_edge
                        map_feature[(ward_feature_indices[ward_type],) + tuple(ward_sector.astype(int))] += 1
        assert map_feature.min() >= 0
        timestep_state['map'] = {'global': map_feature.tolist()}


# Not a general flatten function, assumes a regular shape (tensor)
def flatten_tensor(l: List) -> List:
    while isinstance(l[0], list):
        l = [item for sublist in l for item in sublist]
    return l


def get_champion_state_features(champion_state: List[List],
                                buff_state: List[List],
                                position_index: int,
                                champion_name_index: int,
                                ultimate_name_index: int) -> List[List]:
    blocked_idx = (position_index, champion_name_index, ultimate_name_index)
    # Removes blocked indices from the champion state, adds team feature
    filtered_champion_state = []

    for champ_idx, sublist in enumerate(champion_state):
        new_sublist = []
        for i, item in enumerate(sublist):
            if i not in blocked_idx:
                new_sublist.append(item)
        new_sublist.append(champ_idx >= 5)
        filtered_champion_state.append(new_sublist)

    champion_positions = [[sublist[position_index][coord] for coord in ('x', 'z')] for sublist in champion_state]
    return [state + position + buff
            for state, position, buff in zip(filtered_champion_state, champion_positions, buff_state)]


def get_item_features(items: List[List],
                      item_id_index) -> List[List]:
    return [[[item for i, item in enumerate(item) if i != item_id_index] for item in champion]
            for champion in items]


def get_summoner_spell_features(summoner_spells: List[List],
                                summoner_spell_name_idx: int) -> List[List]:
    return [[[item for i, item in enumerate(summoner_spell) if i != summoner_spell_name_idx] for summoner_spell in champion]
            for champion in summoner_spells]


def get_skill_features(skills: List[List], skill_slot_idx: int) -> List[List]:
    skill_features = [[[item for i, item in enumerate(skill) if i != skill_slot_idx] for skill in champion]
                      for champion in skills]
    is_ultimate_features = [[[skill[skill_slot_idx] == 4] for skill in champion]
                           for champion in skills]
    return [[skill_feature + is_ultimate_feature
             for skill_feature, is_ultimate_feature in zip(champion_skill_features, champion_is_ultimate_features)]
            for champion_skill_features, champion_is_ultimate_features in zip(skill_features, is_ultimate_features)]


# Select features to use
def create_samples(game: dict,
                   champion_to_id: dict,
                   item_to_id: dict,
                   summoner_spell_to_id: dict,
                   skill_to_id: dict,
                   champion_skills: dict) -> List[dict]:
    format = game['format']

    # Categorical indices which need special processing
    pos_idx = format['state']['champion_state'].index('position')
    champion_name_idx = format['state']['champion_state'].index('championName')
    ultimate_name_idx = format['state']['champion_state'].index('ultimateName')
    summoner_spell_name_idx = format['state']['summoner_spells'].index('summonerSpellName')
    item_id_idx = format['state']['items'].index('itemID')
    skill_slot_idx = format['state']['skills'].index('skillSlot')

    champion_names = [champion_state[champion_name_idx].lower() for champion_state in game['data'][0]['state']['champion_state']]
    skill_names = [champion_skills[champion_name] for champion_name in champion_names]
    # Assume champions and their skills do not change over time
    champion_types = [champion_to_id[champion_name] for champion_name in champion_names]
    skill_types = [[skill_to_id[skill] for skill in skills] for skills in skill_names]

    transformed_states = []
    for timestep_data in game['data']:

        timestep_state = timestep_data['state']
        state = {'gameTime': [timestep_data['gameTime']],
                 'map': timestep_state['map']['global'],
                 'state': flatten_tensor(timestep_state['team_building_respawns']) +
                          timestep_state['monster_respawns'] +
                          flatten_tensor(timestep_state['team_buffs']),  # building_respawns, monster_respawns, team_buff_state
                 'stats': flatten_tensor(timestep_state['team_stats']),  # team_statistics
                 # added fog_of_war as last in champion_state
                 'champion_state': get_champion_state_features(timestep_state['champion_state'],
                                                               timestep_state['buffs'],
                                                               pos_idx, champion_name_idx,
                                                               ultimate_name_idx),  # champion_state, buff_state
                 'champion_stats': timestep_state['champion_stats'],  # champion_statistics
                 'items': get_item_features(timestep_state['items'], item_id_idx),  # item_state
                 'control_wards': timestep_state['control_wards'],  # control_ward_state
                 'sight_wards': timestep_state['sight_wards'],  # sight_ward_state
                 'farsight_wards': timestep_state['farsight_wards'],  # farsight_ward_state
                 'summoner_spells': get_summoner_spell_features(timestep_state['summoner_spells'], summoner_spell_name_idx),  # summoner_spell_state
                 'skills': get_skill_features(timestep_state['skills'], skill_slot_idx),  # skill_state
                 # embeddings
                 'champion_types': champion_types,
                 'item_types': [[item_to_id[item[item_id_idx]] for item in sublist]
                                for sublist in timestep_state['items']],
                 'summoner_spell_types': [[summoner_spell_to_id[summoner_spell[summoner_spell_name_idx].lower()]
                                           for summoner_spell in summoner_spells]
                                          for summoner_spells in timestep_state['summoner_spells']],
                 'skill_types': skill_types
                 }
        for key, value in timestep_data['targets'].items():
            state[f'target_{key}'] = value
        transformed_states.append(state)
    return transformed_states


def main() -> None:
    DISCOUNT_FACTOR = 0.7
    TIME_HORIZON = 6
    MAP_SPLIT_N = 16
    POSITON_PREDICTION_SPLIT_N = 12

    #data_folder = sys.argv[1]
    
    data_folder = 'C:\\Users\\edems\\Documents\\Work\\fow_AI\\Game_Data'
    aggregated_folder = os.path.join(data_folder, 'aggregated_fow_data')
    output_folder = os.path.join(data_folder, 'macro_fow')

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    #version = '11.15'
    version = '13.4'
    data_dragon = DataDragon(data_folder, version)
    champion_data = data_dragon.get_champion_data()
    item_data = data_dragon.get_item_data()
    summoner_data = data_dragon.get_summoner_data()

    champion_names = data_dragon.get_champion_names()
    item_ids = list(sorted(item_data['data'].keys()))
    summoner_spell_names = list(sorted(map(str.lower, summoner_data['data'].keys()))) + EXTRA_SUMMONER_SPELL_NAMES
    champion_skill_names = {champion.lower(): [spell['id'].lower() for spell in data['spells']]
                            for champion, data in champion_data['data'].items()}

    # Reserve 0 for unknown
    champion_to_id = {champion_name: i + 1 for i, champion_name in enumerate(champion_names)}
    item_to_id = {int(item_id): i + 1 for i, item_id in enumerate(item_ids)}
    item_to_id[0] = 0
    summoner_spell_to_id = {summoner_spell_name: i + 1 for i, summoner_spell_name in enumerate(summoner_spell_names)}

    skill_to_id = {skill_name: i + 1 for i, skill_name in
                        enumerate([skill for skill_set in champion_skill_names.values() for skill in skill_set])}

    for id, game in tqdm(load_game_data(aggregated_folder)):
        if game['data'][-1]['gameTime'] < 60 * 10 * 1000:  # Assume that games that ended sooner are broken
            print(f"Game {id} was too short, it was ignored.")
            continue
        add_targets(game, POSITON_PREDICTION_SPLIT_N, DISCOUNT_FACTOR, TIME_HORIZON)
        add_features(game, MAP_SPLIT_N)
        for timestep_data in game['data']:
            timestep_data.pop('events')
        samples = create_samples(game, champion_to_id, item_to_id, summoner_spell_to_id, skill_to_id, champion_skill_names)
        with open(os.path.join(output_folder, f'{id}.jsonl'), 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')


if __name__ == '__main__':
    main()
