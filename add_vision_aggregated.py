import os
import json
import numpy as np
import vision_extras as ve
import constants as const

def get_positions(dict):
    return np.array([item['position'] for key in dict for item in dict[key]])


def is_in_vision(player_coords, enemy_players_coords, enemy_buildings_coords, enemy_wards_coords, enemy_minions_coords):

    enemy_players_coords = np.array([coord[:2] for coord in enemy_players_coords])
    player_coords = np.array(player_coords[:2])
    
    enemy_buildings_coords = np.array(enemy_buildings_coords)
    enemy_wards_coords = [np.array([enemy_ward[0][0], enemy_ward[0][1]]) for enemy_ward in enemy_wards_coords]
    
    enemy_minions_coords = get_positions(enemy_minions_coords)
    
    def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    for enemy_player_coords in enemy_players_coords:
        if euclidean_distance(player_coords, enemy_player_coords) <= 1350:
            return True

    for enemy_building_coords in enemy_buildings_coords:
        if euclidean_distance(player_coords, enemy_building_coords) <= 1350:
            return True

    for enemy_ward_coords in enemy_wards_coords:
        if euclidean_distance(player_coords, enemy_ward_coords) <= 900:
            return True

    for enemy_minion_coords in enemy_minions_coords:
        if euclidean_distance(player_coords, enemy_minion_coords) <= 1200:
            return True

    return False



def get_objects_coordinates(team,buildings):
    
    
    enemy_buildings_coords = []
    for lane in buildings[team].values():
        for building in lane:
            if building["buildingType"] == "turret":
                if team == 'teamOne':
                    coords = const.TEAM1_BUILDINGS_DICT[building["lane"]][building["turretTier"]]
                else:
                    coords = const.TEAM2_BUILDINGS_DICT[building["lane"]][building["turretTier"]]
            elif building["buildingType"] == "inhibitor":
                if team == 'teamOne':
                    coords = const.TEAM1_BUILDINGS_DICT["inhibitor"][building["lane"]]
                else:
                    coords = const.TEAM2_BUILDINGS_DICT["inhibitor"][building["lane"]]
            enemy_buildings_coords.append(coords)

    return np.array(enemy_buildings_coords)





def main()->None:

    # The path to the directory with aggregated JSON files
    base_dir = r"C:\Users\edems\Work\Edit_fantasy_AI_muller\Game_data"
    sub_dir = "aggregated"
    dir_path = os.path.join(base_dir, sub_dir)

    filenames = []

    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):
            filenames.append(filename)

    if filenames:
        for filename in filenames:
            first_file_path = os.path.join(dir_path, filename)

            with open(first_file_path, 'r') as file:
                    curr_json = json.load(file)
            
            curr_json['format']['state']['champion_state'].append('fog_of_war')
            
            buildings = ve.get_static_buildings()
            current_minions = {}
            FIRST_SPAWN = False
            minion_spawn_addition = 30000
            minion_spawn = 65000
            
            
            for row in curr_json['data']:
                vision_status_t1,vision_status_t2 = [],[]
                player_position_t1,wards_t1 = [],[]
                player_position_t2,wards_t2 = [],[]
                
                game_time = row['gameTime']
                
                # CHAMPION POSITIONS
                player_position_t1, player_position_t2 = ve.parse_champion_positions(row)

                
                # WARDS 
                wards_t1, wards_t2 = ve.parse_wards(row)
                
                # BUILDINGS
                for event in row['events']:
                    if event[0] == 'building_destroyed':
                        buildings = ve.delete_building(game_time,buildings,event)
                
                # MINIONS
                current_minions = ve.delete_minions(current_minions,buildings,game_time)
                current_minions = ve.update_minions(game_time,current_minions,buildings)
                
                if game_time>= minion_spawn:
                    if FIRST_SPAWN == False:
                        FIRST_SPAWN = True
                        current_minions = ve.spawn_minions(current_minions,game_time)
                        continue
                    minion_spawn +=minion_spawn_addition
                    current_minions = ve.spawn_minions(current_minions,game_time)

                enemy_buildings_coords_t2 = get_objects_coordinates(team=const.TeamKeys.t1.value,buildings=buildings)
                
                for player in player_position_t1:
                    vision_status = is_in_vision(
                        player_coords=player,
                        enemy_players_coords=player_position_t2,
                        enemy_buildings_coords=enemy_buildings_coords_t2,
                        enemy_wards_coords=wards_t2,
                        enemy_minions_coords=current_minions.get('teamTwo', []),
                    )
                    vision_status_t1.append(vision_status)
                    
                enemy_buildings_coords_t1 = get_objects_coordinates(team=const.TeamKeys.t2.value,buildings=buildings)
                
                for player in player_position_t2:
                    vision_status = is_in_vision(
                        player_coords=player,
                        enemy_players_coords=player_position_t1,
                        enemy_buildings_coords=enemy_buildings_coords_t1,
                        enemy_wards_coords=wards_t1,
                        enemy_minions_coords=current_minions.get('teamOne', []),
                    )
                    vision_status_t2.append(vision_status)
                
                for index,vision in enumerate(vision_status_t1):
                    row['state']['champion_state'][index].append(vision) 
                
                for index,vision in enumerate(vision_status_t1,5):
                    row['state']['champion_state'][index].append(vision) 

                
            save_dir = "aggregated_fow_data"
            full_path = os.path.join(base_dir,save_dir,filename)
            
            with open(full_path, 'w') as fileout:
                json.dump(curr_json, fileout, indent=4)
    
if __name__ == '__main__':
    main()