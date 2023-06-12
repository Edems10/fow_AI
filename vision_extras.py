from ast import List
from enum import Enum
import constants as const
import numpy as np

class BuildingType(Enum):
    turret =1
    inhibitor =2
    
class TurretTier(Enum):
    nexus_top = 'nexus_top'
    nexus_bot = 'nexus_bot'
    turret_base = 'turret_base'
    turret_inner = 'turret_inner'
    turret_outer = 'turret_outer'

class TeamType(Enum):
    teamOne = 1
    teamTwo = 2
    
class Lane(Enum):
    top  = 1
    mid  = 2
    bot  = 3 


class Minion:
    distance_traveled:float
    position:List
    created:int
    
    
    def __init__(self,dt,pos,created):
        self.distance_traveled = dt
        self.position = pos
        self.created = created
        
    def update(self,dt,pos):
        self.distance_traveled = dt
        self.position = pos

def parse_wards(row):
    wards_t1 = []
    wards_t2 = []
    player_iter = 0

    for ward_type in ['sight_wards', 'farsight_wards', 'control_wards']:
        for ward_pos in row['state'][ward_type]:
            if player_iter < 5:
                for ward in ward_pos:
                    if ward[0] != 0 and ward[1] != 0:
                        wards_t1.append([ward])
            else:
                for ward in ward_pos:
                    if ward[0] != 0 and ward[1] != 0:
                        wards_t2.append([ward])
            player_iter += 1
        player_iter = 0

    return wards_t1, wards_t2

def delete_building(game_time,buildings,event)->dict:

        event = event[1]
        lane  = event[3]
        if event[1] == 200:
            team = TeamType.teamOne.name
        else:
            team = TeamType.teamTwo.name
        building_type = event[4]

        if building_type in (item.value for item in TurretTier)and event[4] != 'nexus':
            turret_tier = event[4]
            for team_selected, lanes_selected in buildings.items():
                if team_selected == team:
                    for selected_lane, selected_buildings in lanes_selected.items():
                        if selected_lane == lane:
                            for building in selected_buildings:
                                if building['turretTier']== turret_tier:
                                    selected_buildings.remove(building)
                                    return buildings                        
        
        elif building_type == BuildingType.inhibitor.name:
            for team_selected, lanes_selected in buildings.items():
                if team_selected == team:
                    for selected_lane, selected_buildings in lanes_selected.items():
                        if selected_lane == BuildingType.inhibitor.name:
                            for building in selected_buildings:
                                if building['lane']== lane:
                                    update_inhibitor = building
                                    update_inhibitor['destroyed']=game_time
                                    selected_buildings.remove(building)
                                    selected_buildings.append(update_inhibitor)
                                    return buildings
        
        else:
            position = [event[5]['x'],event[5]['z']]
            for team_selected, lanes_selected in buildings.items():
                if team_selected == team:
                    for selected_lane, selected_buildings in lanes_selected.items():
                        if selected_lane == lane:
                            for building in selected_buildings:
                                turret_tier = building['turretTier']
                                if team == building['team']:
                                    if position == building['position']:
                                        selected_buildings.remove(building)
                                        return buildings
        return "gas"

def get_turrets(building_type,turret_tier,lane,team)->dict:
    if turret_tier != TurretTier.nexus_bot.name and turret_tier != TurretTier.nexus_top.name:
        return  {"buildingType":building_type,
                            "turretTier":turret_tier,
                            "team":team,
                            "lane":lane}
    
    if team == TeamType.teamOne.name:
        if turret_tier == TurretTier.nexus_bot.name:
            position = const.TURRET_MID_BOT_NEXUS_TEAM1
        elif turret_tier == TurretTier.nexus_top.name:
            position = const.TURRET_MID_TOP_NEXUS_TEAM1
    elif team == TeamType.teamTwo.name:
        if turret_tier == TurretTier.nexus_bot.name:
            position = const.TURRET_MID_BOT_NEXUS_TEAM2
        elif turret_tier == TurretTier.nexus_top.name:
            position = const.TURRET_MID_TOP_NEXUS_TEAM2

    
    return  {"buildingType":building_type,
                            "turretTier":turret_tier,
                            "team":team,
                            "lane":lane,
                            'position':position}

def get_inhibitor(building_type,lane,team):
    return  {"buildingType":building_type,
                        "team":team,
                        "lane":lane,
                        'destroyed':False}

def get_static_buildings()->dict:
    """
    Returns a dictionary of static buildings for each team in the game.
    Each team's buildings are organized by lane and type (top, mid, bottom, inhibitor).

    Returns:
    dict: A dictionary of static buildings for each team in the game.
            The dictionary is organized as follows:
            {
                'TeamOne': {
                    'top': [list of top turrets],
                    'mid': [list of mid turrets],
                    'bottom': [list of bottom turrets],
                    'inhibitor': [list of inhibitors]
                },
                'TeamTwo': {
                    'top': [list of top turrets],
                    'mid': [list of mid turrets],
                    'bottom': [list of bottom turrets],
                    'inhibitor': [list of inhibitors]
                }
            }
    """
    buildings_dict = {}
    bot_top_towers_tier_list = [TurretTier.turret_base.name,TurretTier.turret_inner.name,TurretTier.turret_outer.name]
    
    for team in TeamType:
        top_turrets,mid_turrets,bottom_turrets,inhibitors = [], [], [], []
        
        for tier in bot_top_towers_tier_list:
            turret_top_base = get_turrets(building_type=BuildingType.turret.name,
                                            team=team.name,
                                            lane=Lane.top.name,
                                            turret_tier=tier)
            top_turrets.append(turret_top_base)
        
        for tier in bot_top_towers_tier_list:
            turret_top_base = get_turrets(building_type=BuildingType.turret.name,
                                            team=team.name,
                                            lane=Lane.bot.name,
                                            turret_tier=tier)
            bottom_turrets.append(turret_top_base)
        
        for tier in TurretTier:
            turret_mid_base = get_turrets(building_type=BuildingType.turret.name,
                                            team=team.name,
                                            lane=Lane.mid.name,
                                            turret_tier=tier.name)
            mid_turrets.append(turret_mid_base)
        
        for lane in Lane:
        
            inhibitor_top = get_inhibitor(building_type=BuildingType.inhibitor.name,
                                        team=team.name,
                                        lane=lane.name)
            inhibitors.append(inhibitor_top)

            
        team_dict={team.name:{'top':top_turrets,
                            'mid':mid_turrets,
                            'bot':bottom_turrets,
                            'inhibitor':inhibitors}}
        buildings_dict.update(team_dict)
    
    return buildings_dict

def get_traveled_distance(minion,game_time):
    minion_ms = minion_movement_speed(game_time)
    time_traveled = game_time - int(minion['created'])
    return ((time_traveled/1000)*minion_ms)

def minion_movement_speed(game_time):
        bonuses = {
        1500000: 100,
        1200000: 75,
        900000: 50,
        600000: 25}

        bonus = 0
        for threshold, value in bonuses.items():
            if game_time >= threshold:
                bonus = value
            else:
                break
        return const.MINION_BASE_MOVEMENT_SPEED+bonus

def get_position_in_lane(waypoints, distance,max_segment,lane)->list:
    max_segment = 10 - max_segment if lane in [const.Lanes.top.name, const.Lanes.bot.name] else 9 - max_segment
    # Calculate the segment of the line to traverse
    remaining_distance = distance
    for i in range(len(waypoints) - 1):
        if max_segment == 0:
            return waypoints[i + 1]
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        segment_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if remaining_distance >= segment_distance:
            remaining_distance -= segment_distance
            max_segment-=1
        else:
            # Calculate the exact position where the given distance ends
            ratio = remaining_distance / segment_distance
            x = x1 + (x2 - x1) * ratio
            y = y1 + (y2 - y1) * ratio
            return (x, y)

    # If there is still remaining distance, return the last possible waypoint
    return (waypoints[-1][0], waypoints[-1][1])

def update_minions(game_time,minions,buildings)->dict:
    if minions == {}:
        return {}
    for team_name,positions in minions.items():
        for lane,minion_list in positions.items():
            if not minion_list:
                continue
            for i in range(len(minion_list)):
                current_distances = get_traveled_distance(minion_list[i], game_time)
                updated_positions = get_position_in_lane(
                const.MINION_WAYPOINTS[team_name][lane],
                current_distances,
                len(buildings[team_name][lane]),
                lane)
                minion_list[i]['position'] = updated_positions
                minion_list[i]['distance'] = current_distances
            minions[team_name][lane]= minion_list
    return minions

def spawn_minions(current_minions,game_time)->dict:
    for team in const.TeamKeys:
        new_minons = {}            
        for lane in const.Lanes:
            if team.value in current_minions:
                if lane.value in current_minions[team.value]:
                    min_val = [current_minions[team.value][lane.value]]
                    min = Minion(0,const.MINION_WAYPOINTS[team.value][lane.value][0],game_time)
                    min_val[0].append({'position':min.position,'distance':min.distance_traveled,'created':min.created})
                    new_minons[lane.name] = min_val[0]
            else:
                min = Minion(0,const.MINION_WAYPOINTS[team.value][lane.value][0],game_time)
                new_minons[lane.name] = [{'position':min.position,'distance':min.distance_traveled,'created':min.created}]
                
        current_minions[team.value] = new_minons
    return current_minions

def delete_minions(minions,buildings,game_time)->dict:
    if minions == {}:
        return {}
    for team_name,positions in minions.items():
        for lane,minion_list in positions.items():
            if not minion_list:
                continue
            index_to_delete = []
            for i in range(len(minion_list)):
                current_distances = get_traveled_distance(minion_list[i], game_time)
                delete_minion = minion_delete_calculator(
                    const.MINION_WAYPOINTS[team_name][lane],
                    current_distances,
                    len(buildings[team_name][lane]),
                    lane)
                if delete_minion:
                    index_to_delete.append(i)
            for del_index in index_to_delete:
                del minions[team_name][lane][del_index]
    return minions


def minion_delete_calculator(waypoints, distance,max_segment,lane)->bool:
    max_segment = 10 - max_segment if lane in [const.Lanes.top.name, const.Lanes.bot.name] else 9 - max_segment
    # Calculate the segment of the line to traverse
    remaining_distance = distance
    overflow_distance = 0
    for i in range(len(waypoints) - 1):
        if max_segment == 0:
            overflow_distance += remaining_distance
            break
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        segment_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if remaining_distance >= segment_distance:
            remaining_distance -= segment_distance
            max_segment-=1
        else:
            return False
    if overflow_distance >=2000: 
        return True
    else:
        return False 
    
    
def parse_champion_positions(row):
    player_position_t1 = []
    player_position_t2 = []
    player_iter = 0

    for champion_pos in row['state']['champion_state']:
        if player_iter < 5:
            player_position_t1.append([champion_pos[5]['x'], champion_pos[5]['z']])
        else:
            player_position_t2.append([champion_pos[5]['x'], champion_pos[5]['z']])
        player_iter += 1

    return player_position_t1, player_position_t2
