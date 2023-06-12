import os
import pandas as pd
import sys
from tqdm import tqdm

from utils import load_game_data, determine_champion_roles
from lol_rules import CHAMPION_ROLES, WARD_TYPES, DRAGON_TYPES, LANES, TURRET_TYPES


def main() -> None:
    data_folder = 'C:\\Users\\edems\\Work\\Edit_fantasy_AI_muller\\Game_data'
    aggregated_folder = os.path.join(data_folder, 'aggregated')
    output_folder = 'data'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Features per team
    team_features = ['total_gold', 'champion_kills', 'deaths'] + \
                    [f'{role}_level' for role in CHAMPION_ROLES] + \
                    [f'{role}_respawn' for role in CHAMPION_ROLES] + \
                    [f'{role}_champion' for role in CHAMPION_ROLES] + \
                    [f'{type}_dragon_killed' for type in DRAGON_TYPES] + ['baron_kills'] +\
                    ['dragon_buff_remaining', 'baron_buff_remaining'] + \
                    TURRET_TYPES + \
                    [f'inhibitor_{lane}' for lane in LANES] + \
                    [f'inhibitor_{lane}_respawn' for lane in LANES] + \
                    [f'{ward_type}_wards' for ward_type in WARD_TYPES]

    dataset = {}
    for id, game in tqdm(load_game_data(aggregated_folder)):
        format = game['format']
        team_stats_format = format['state']['team_stats']
        game_states = {}
        current_team_features = [[0 for _ in range(len(team_features))] for _ in range(2)]
        champion_roles = determine_champion_roles(game)
        name_index = format['state']['champion_state'].index('championName')
        champions = [champion[name_index] for champion in game['data'][0]['state']['champion_state']]
        winning_team = None
        for timestep_data in game['data']:
            timestep_state = timestep_data['state']
            game_time = timestep_data['gameTime']
            for team in range(2):
                current_team_features[team][team_features.index('total_gold')] =\
                    timestep_state['team_stats'][team][team_stats_format.index('totalGold')]
                current_team_features[team][team_features.index('champion_kills')] =\
                    timestep_state['team_stats'][team][team_stats_format.index('championsKills')]
                current_team_features[team][team_features.index('deaths')] =\
                    timestep_state['team_stats'][team][team_stats_format.index('deaths')]
                current_team_features[team][team_features.index('baron_buff_remaining')] =\
                    timestep_state['team_buffs'][team][format['state']['team_buffs'].index('baron_remaining')]
                current_team_features[team][team_features.index('dragon_buff_remaining')] = \
                    timestep_state['team_buffs'][team][format['state']['team_buffs'].index('dragon_remaining')]
                for role in CHAMPION_ROLES:
                    current_team_features[team][team_features.index(f'{role}_level')] =\
                        timestep_state['champion_state'][champion_roles[team + 1][role]][format['state']['champion_state'].index('level')]
                    current_team_features[team][team_features.index(f'{role}_respawn')] =\
                        timestep_state['champion_state'][champion_roles[team + 1][role]][format['state']['champion_state'].index('respawnTimer')]
                    current_team_features[team][team_features.index(f'{role}_champion')] =\
                        champions[champion_roles[team + 1][role]]
                for ward_type in WARD_TYPES:
                    current_team_features[team][team_features.index(f'{ward_type}_wards')] =\
                        sum(len([w for w in timestep_state[f'{ward_type}_wards'][i] if w[0] != 0 or w[0] != 0])
                            for i in range(5 * team, 5 * (team + 1)))
                for lane in LANES:
                    feature_name = f'inhibitor_{lane}_respawn'
                    current_team_features[team][team_features.index(feature_name)] =\
                        timestep_state['team_building_respawns'][team][format['state']['team_building_respawns'].index(feature_name)]
                dragon_buffs = 0
                for dragon_type in DRAGON_TYPES:
                    if dragon_type != 'elder':
                        current_team_features[team][team_features.index(f'{dragon_type}_dragon_killed')] =\
                            timestep_state['team_buffs'][team][format['state']['team_buffs'].index(dragon_type)]
                        dragon_buffs += current_team_features[team][team_features.index(f'{dragon_type}_dragon_killed')]
                current_team_features[team][team_features.index("elder_dragon_killed")] =\
                    timestep_state['team_stats'][team][team_stats_format.index('dragonKills')] - dragon_buffs
                current_team_features[team][team_features.index('baron_kills')] =\
                    timestep_state['team_stats'][team][team_stats_format.index('baronKills')]
            for event_type, event in timestep_data['events']:
                if event_type == 'building_destroyed':
                    killer_team_index = 2 - event[format['events']['building_destroyed'].index('teamID')] // 100
                    building_type = event[format['events']['building_destroyed'].index('buildingType')]
                    lane = event[format['events']['building_destroyed'].index('lane')]
                    if 'turret' in building_type or building_type == 'inhibitor':
                        current_team_features[killer_team_index][team_features.index(f"{building_type}_{lane}")] += 1
                elif event_type == 'game_end':
                    winning_team = event[format['events']['game_end'].index('winningTeam')] // 100 - 1

            # Add current state to dataset
            game_states[game_time] = current_team_features[0] + current_team_features[1]
        if winning_team is not None:
            dataset[id] = {"states": game_states, "winner": winning_team}
    df = pd.DataFrame(data=[[id, timestep, dataset[id]['winner']] + dataset[id]['states'][timestep]
                            for id in dataset for timestep in sorted(dataset[id]['states'])],
                      columns=['id', 'time', 'winner'] + [f't1_{f}' for f in team_features] + [f't2_{f}' for f in team_features])
    df.to_csv(os.path.join(output_folder, 'win_dataset.csv'))


if __name__ == '__main__':
    main()
