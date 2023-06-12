import os
import sys
from collections import defaultdict
from tqdm import tqdm

from utils import load_game_data

# Output of main
BUILDING_POSITONS = {'turret_outer_1': {(4318, 13875), (13866, 4505), (8955, 8510)},
                     'turret_inner_1': {(13327, 8226), (9767, 10113), (7943, 13411)},
                     'turret_base_1': {(10481, 13650), (11134, 11207), (13624, 10572)},
                     'inhibitor_1': {(11261, 13676), (13604, 11316), (11598, 11667)},
                     'turret_nexus_1': {(12611, 13084), (13052, 12612)},
                     'turret_outer_0': {(5846, 6396), (981, 10441), (10504, 1029)},
                     'turret_inner_0': {(6919, 1483), (5048, 4812), (1512, 6699)},
                     'turret_base_0': {(4281, 1253), (3651, 3696), (1169, 4287)},
                     'inhibitor_0': {(3203, 3208), (3452, 1236), (1171, 3571)},
                     'turret_nexus_0': {(1748, 2270), (2177, 1807)}}


def main() -> None:
    data_folder = sys.argv[1]
    aggregated_folder = os.path.join(data_folder, 'aggregated')
    building_positions = defaultdict(lambda: set())
    for id, game in tqdm(load_game_data(aggregated_folder)):
        format = game['format']
        building_type_index = format['events']['building_destroyed'].index('buildingType')
        # lane_index = format['events']['building_destroyed'].index('lane')
        team_id_index = format['events']['building_destroyed'].index('teamID')
        position_index = format['events']['building_destroyed'].index('position')
        for timestep_data in game['data']:
            for event_type, event in timestep_data['events']:
                if event_type == 'building_destroyed':
                    building_type = event[building_type_index]
                    # lane = event[lane_index]
                    team_id = event[team_id_index] // 100 - 1
                    position = event[position_index]
                    building_positions[f"{building_type}_{team_id}"].add((position['x'], position['z']))
    print(building_positions)
    # for key, counter in building_positions.items():
    #     assert(len(counter)) == 1
    # print({key: value[value.keys()[0]] for key, value in building_positions.items()})


if __name__ == '__main__':
    main()
