import json
from collections import Counter

from tqdm import tqdm

from utils import load_game_data


def main() -> None:
    for id, game in tqdm(load_game_data('D:/fantasyai/data/aggregated')):
        turret_types = Counter()
        game_turrets_destroyed = [0, 0]
        format = game['format']
        # summoner_spell_name_index = format['state']['summoner_spell_sets'].index('summonerSpellName')
        building_type_index = format['events']['building_destroyed'].index('buildingType')
        lane_index = format['events']['building_destroyed'].index('lane')
        team_id_index = format['events']['building_destroyed'].index('teamID')
        team_stats_towers_destroyed_index = format['state']['team_stats'].index('towerKills')
        for timestep_data in game['data']:
            for event_type, event in timestep_data['events']:
                if event_type == 'building_destroyed' and 'turret' in event[building_type_index]:
                    turret_types[f"{event[building_type_index]}_{event[lane_index]}"] += 1
                    game_turrets_destroyed[event[team_id_index] // 100 - 1] += 1
            if 'team_stats' in timestep_data['state']:
                team_tower_kills = [timestep_data['state']['team_stats'][i][team_stats_towers_destroyed_index] for i in range(2)]
                if max(team_tower_kills) > max(game_turrets_destroyed):
                    champion_names = [champion_state[format['state']['champion_state'].index('championName')] for champion_state in timestep_data['state']['champion_state']]
                    if 'Azir' not in champion_names:
                        print(id, timestep_data['gameTime'], team_tower_kills, game_turrets_destroyed, turret_types)
        if max(game_turrets_destroyed) >= 12:
            print(id, game_turrets_destroyed, turret_types)


if __name__ == '__main__':
    main()
