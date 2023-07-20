import copy
import json
import numpy as np
import os
import sys
from tqdm import tqdm
from typing import List


from utils import read_matches_from_batches
from lol_rules import DRAGON_TYPES, LANES, TEAM_EPIC_MONSTERS, NEUTRAL_EPIC_MONSTERS, INHIBITOR_RESPAWN_TIME, \
    RESPAWN_TIMES, SPAWN_TIMES, CAMP_BUFF_DURATIONS, BARON_BUFF_DURATION, DRAGON_BUFF_DURATION, \
    WARD_TYPE_MAP, WARD_DURATIONS, WARD_LIMITS
from data_dragon import DataDragon


def update_timestep_data_timers(format: dict, timestamp_data: dict, interval: int) -> None:
    # Update remaining ward timers
    timeleft_index = format['state']['sight_wards'].index('timeLeft')
    for i in range(10):
        sight_wards = timestamp_data['state']['sight_wards'][i]
        for sight_ward in sight_wards:
            sight_ward[timeleft_index] = max(sight_ward[timeleft_index] - interval, 0)
            if sight_ward[timeleft_index] <= 0:
                sight_ward[:2] = 0, 0


def update_timestep_data(timestep_data: dict,
                         current_timestep: int,
                         inhibitor_destroyed_times: np.ndarray,
                         monster_killed_times: np.ndarray,
                         monster_respawn_vector: np.ndarray,
                         monster_spawn_vector: np.ndarray,
                         buff_acquired_times: np.ndarray,
                         baron_killed_times: np.ndarray,
                         elder_dragon_killed_times: np.ndarray,
                         team_buff_state: List[str],
                         monster_respawns: List[str]) -> None:
    timestep_data['gameTime'] = current_timestep
    # Updated respawns based on death timers and current timestamp
    timestep_data['state']['team_building_respawns'] = np.maximum(
        inhibitor_destroyed_times + INHIBITOR_RESPAWN_TIME - current_timestep, 0).tolist()
    # Note that Rift herald's respawn timer after the 20-minute mark doesn't matter
    timestep_data['state']['monster_respawns'] = np.maximum(
        monster_killed_times + monster_respawn_vector - current_timestep,
        np.maximum(monster_spawn_vector - current_timestep, 0)
    ).tolist()
    # Add remaining time until next minion respawn
    minion_index = monster_respawns.index('minions')
    if monster_spawn_vector[minion_index] < current_timestep:
        timestep_data['state']['monster_respawns'][minion_index] = \
            (current_timestep - monster_spawn_vector[minion_index].item()) % monster_respawn_vector[minion_index].item()
    timestep_data['state']['buffs'] = np.maximum(
        CAMP_BUFF_DURATIONS + buff_acquired_times - current_timestep, 0).tolist()
    for team_index in range(2):
        timestep_data['state']['team_buffs'][team_index][team_buff_state.index('baron_remaining')] = \
            np.maximum(baron_killed_times[team_index] + BARON_BUFF_DURATION - current_timestep, 0).item()
        timestep_data['state']['team_buffs'][team_index][team_buff_state.index('dragon_remaining')] = \
            np.maximum(elder_dragon_killed_times[team_index] + DRAGON_BUFF_DURATION - current_timestep, 0).item()


# FIXME: In dire need of refactoring (move aggregation to a class)
def main() -> None:

    # Joins the current directory with the relative directory 'Game_data'
    #data_folder = 'C:\\Users\\edems\\Work\\mulleste_master_code\\DATA_FOLDER_REMADE_TEST'
    data_folder = 'C:\\Users\\edems\\Documents\\Work\\fow_AI\\Game_Data'
    version = '13.4'
    #version = '11.15'
    
    interval = 1000
    # interpolate = False

    data_dragon = DataDragon(data_folder, version)
    item_data = data_dragon.get_item_data()
    for itemID, item in item_data['data'].items():
        if item['name'] == 'Vigilant Wardstone':
            vigilant_wardstone_id = itemID
            break
    else:
        raise ValueError("Vigilant Wardstone item id not found!")

    output_folder = os.path.join(data_folder, 'aggregated')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    team_statistics = ['towerKills', 'assists', 'inhibKills', 'totalGold', 'championsKills', 'deaths', 'dragonKills',
                       'baronKills']

    champion_state = ['championName',
                      'alive',
                      'respawnTimer',
                      'level',
                      'XP',
                      'position',
                      'health', 'healthMax', 'healthRegen',
                      'magicPenetration', 'magicPenetrationPercent', 'magicPenetrationPercentBonus',
                      'armorPenetration', 'armorPenetrationPercent', 'armorPenetrationPercentBonus',
                      'currentGold', 'totalGold', 'goldPerSecond',
                      'shutdownValue',
                      'primaryAbilityResource', 'primaryAbilityResourceMax', 'primaryAbilityResourceRegen',
                      'attackDamage', 'attackSpeed',
                      'abilityPower', 'cooldownReduction',
                      'lifeSteal', 'spellVamp',
                      'armor', 'magicResist',
                      'ccReduction',
                      'ultimateName', 'ultimateCooldownRemaining'
                      ]
    building_respawns = [f'inhibitor_{lane}_respawn' for lane in LANES]
    monster_respawns = ['minions'] +\
                       [f't{i}_{team_monster}' for i in range(1, 3) for team_monster in TEAM_EPIC_MONSTERS] +\
                       NEUTRAL_EPIC_MONSTERS
    buff_state = ['red_remaining', 'blue_remaining']
    team_buff_state = ['baron_remaining', 'dragon_remaining'] + [dragon_type for dragon_type in DRAGON_TYPES if dragon_type != 'elder']
    champion_statistics = ['minions_killed', 'neutral_minions_killed',
                           'neutral_minions_killed_your_jungle', 'neutral_minions_killed_enemy_jungle',
                           'champions_killed', 'num_deaths', 'assists',
                           'ward_placed', 'ward_killed',
                           'vision_score',
                           # Riot stats
                           'total_damage_dealt',
                           'physical_damage_dealt_player', 'magic_damage_dealt_player', 'true_damage_dealt_player',
                           'total_damage_dealt_to_champions', 'physical_damage_dealt_to_champions',
                           'magic_damage_dealt_to_champions', 'true_damage_dealt_to_champions',
                           'total_damage_taken', 'physical_damage_taken', 'magic_damage_taken', 'true_damage_taken',
                           'total_damage_self_mitigated', 'total_damage_shielded_on_teammates',
                           'total_damage_dealt_to_buildings',
                           'total_damage_dealt_to_turrets',
                           'total_damage_dealt_to_objectives',
                           'total_time_crowd_control_dealt',
                           'total_heal_on_teammates',
                           'time_ccing_others']

    # Sets
    item_state = ['itemID', 'itemCooldown']  # <= item_purchased, item_destroyed, item_sold and item_undo events
    sight_ward_state = ['x', 'y', 'timeLeft']  # <= placed_ward and killed_ward events
    control_ward_state = ['x', 'y']  # <= placed_ward and killed_ward events
    farsight_ward_state = ['x', 'y']  # <= placed_ward and killed_ward events
    summoner_spell_state = ['summonerSpellName', 'summonerSpellCooldownRemaining']
    skill_state = ['skillSlot', 'level', 'evolved']  # <= skill_level_up event
    perk_state = ['perkID', 'var1', 'var2', 'var3']

    # Events
    # Item events could be redundant because of stats_update
    item_purchased = ['participantID', 'itemID']
    item_destroyed = ['participantID', 'itemID']
    item_sold = ['participantID', 'itemID']
    item_undo = ['participantID', 'itemID', 'goldGain']
    champion_level_up = ['participant', 'level']  # redundant because of stats_update?
    ward_placed = ['wardType', 'placer', 'position']
    ward_kill = ['wardType', 'killer', 'position']
    epic_monster_kill = ['assistants', 'inEnemyJungle', 'monsterType', 'killer', 'position']
    champion_kill = ['assistants', 'bounty', 'killStreakLength', 'victim', 'killer', 'position']
    building_destroyed = ['assistants', 'teamID', 'lastHitter', 'lane', 'buildingType', 'position']
    turret_plate_destroyed = ['assistants', 'teamID', 'lastHitter', 'lane', 'position']
    champion_kill_special = ['killType', 'killer', 'position']
    epic_monster_spawn = ['dragonType', 'monsterType']  # redundant because of queued_dragon_info?
    queued_dragon_info = ['nextDragonName', 'nextDragonSpawnTime']
    game_end = ['winningTeam']
    quit = ['participant']
    reconnect = ['participant']
    champion_transformed = ['transformer', 'transformType']

    format = {'state': {'team_stats': team_statistics,  # array[2]
                        'champion_state': champion_state,  # array[10]
                        'team_building_respawns': building_respawns,  # array[2]
                        'monster_respawns': monster_respawns,  # flat
                        'buffs': buff_state,  # array[10]
                        'team_buffs': team_buff_state,  # array[2]
                        'champion_stats': champion_statistics,  # array[10]
                        'items': item_state,  # array[10, 7]
                        'sight_wards': sight_ward_state,  # array[10, 4]
                        'control_wards': control_ward_state,  # array[10, 2]
                        'farsight_wards': farsight_ward_state,  # array[10, 20]
                        'summoner_spells': summoner_spell_state,  # array[10, 2]
                        'skills': skill_state,  # array[10, 4]
                        'perks': perk_state},  # array[10, 6]
              'events': {'item_purchased': item_purchased,
                         'item_destroyed': item_destroyed,
                         'item_sold': item_sold,
                         'item_undo': item_undo,
                         'champion_level_up': champion_level_up,
                         'ward_placed': ward_placed,
                         'ward_killed': ward_kill,
                         'epic_monster_kill': epic_monster_kill,
                         'champion_kill': champion_kill,
                         'building_destroyed': building_destroyed,
                         'turret_plate_destroyed': turret_plate_destroyed,
                         'champion_kill_special': champion_kill_special,
                         'epic_monster_spawn': epic_monster_spawn,
                         'queued_dragon_info': queued_dragon_info,
                         'game_end': game_end,
                         'quit': quit,
                         'reconnect': reconnect,
                         'champion_transformed': champion_transformed}}

    # Track how many times were we able to match ward kill with ward placements
    ward_kill_attempts, ward_kill_successes = 0, 0

    for events in tqdm(read_matches_from_batches(data_folder, version)):
        data = {'format': format,
                "data": []}

        timestep_data = {'state': {'team_building_respawns': [[0] * len(building_respawns) for _ in range(2)],
                                   'monster_respawns': [0] * len(monster_respawns),
                                   'buffs': [[0] * len(buff_state) for _ in range(10)],
                                   'team_buffs': [[0] * len(team_buff_state) for _ in range(2)],
                                   'skills': [[[i+1, 0, 0] for i in range(4)] for _ in range(10)],
                                   'items': [[[0, 0] for _ in range(7)] for _ in range(10)],
                                   'sight_wards': [[[0, 0, 0] for _ in range(4)] for _ in range(10)],
                                   'control_wards': [[[0, 0] for _ in range(2)] for _ in range(10)],
                                   'farsight_wards': [[[0, 0] for _ in range(20)] for _ in range(10)]},
                         'events': []}
        baron_killed_times = np.full(2, -np.inf)
        elder_dragon_killed_times = np.full(2, -np.inf)
        inhibitor_destroyed_times = np.full((2, 3), -np.inf)
        monster_killed_times = np.full(len(monster_respawns), -np.inf)
        buff_acquired_times = np.full((10, 2), -np.inf)
        dragon_type = None
        crabs_killed = 0  # Scuttle crab starts respawning after the first 2 are killed
        dragons_killed = np.zeros(2)  # Elder dragon spawns after a team kills 4 dragons
        monster_spawn_vector = np.array([SPAWN_TIMES[monster.split('_')[-1]] for monster in monster_respawns])
        monster_respawn_vector = np.array(
            [RESPAWN_TIMES[monster.split('_')[-1]] for monster in monster_respawns])

        current_timestep = interval
        for event in events:
            # Next timestamp reached, append data
            if event['sequenceIndex']<=0:
                continue
            while event['gameTime'] > current_timestep:
                if 'team_stats' in timestep_data['state']:  # Only save state if we have seen a stats_update event
                    update_timestep_data(timestep_data,
                                         current_timestep,
                                         inhibitor_destroyed_times,
                                         monster_killed_times,
                                         monster_respawn_vector,
                                         monster_spawn_vector,
                                         buff_acquired_times,
                                         baron_killed_times,
                                         elder_dragon_killed_times,
                                         team_buff_state,
                                         monster_respawns)
                    data['data'].append(copy.deepcopy(timestep_data))
                timestep_data['events'].clear()
                current_timestep += interval
                update_timestep_data_timers(format, timestep_data, interval)

            schema = event['rfc461Schema']
            if schema == 'stats_update':
                timestep_data['state']['team_stats'] = [[event['teams'][i][key] for key in team_statistics]
                                                        for i in range(2)]
                participants = event['participants']
                timestep_data['state']['champion_state'] = [[participants[i][key] for key in champion_state]
                                                            for i in range(10)]
                # FIXME: Inefficient O(N^2) but maintains order
                timestep_data['state']['champion_stats'] = [[[item for item in participants[i]['stats']
                                                             if item['name'].lower() == stat][0]['value']
                                                             for stat in champion_statistics]
                                                             for i in range(10)]

                # The itemCooldown and participant.items keys are sometimes missing
                timestep_data['state']['items'] = [[[item['itemID'], item['itemCooldown'] if 'itemCooldown' in item else 0]
                                                     for item in participants[i]['items']] +
                                                     [[0] * len(item_state) for _ in range(7 - len(participants[i]['items']))]
                                                     if 'items' in participants[i] else timestep_data['state']['items'][i]
                                                     for i in range(10)]
                timestep_data['state']['summoner_spells'] = [[[participants[i][f'summonerSpell{j}Name'],
                                                               participants[i][f'summonerSpell{j}CooldownRemaining']]
                                                               for j in range(1, 3)]
                                                               for i in range(10)]
                # FIXME: Inefficient O(N^2) but maintains order
                timestep_data['state']['perks'] = [[[[item for item in participants[i]['stats']
                                                        if item['name'] == f'PERK{j}{f"_VAR{var}" if var else ""}'][0]['value']
                                                        for var in range(4)]
                                                        for j in range(6)]
                                                        for i in range(10)]
            elif schema == 'skill_level_up':
                skill = timestep_data['state']['skills'][event['participant'] - 1][event['skillSlot'] - 1]
                skill[1] += 1
                skill[2] = event['evolved']
            elif schema in format['events']:
                if schema == 'building_destroyed' and 'turretTier' in event:
                    event['buildingType'] += "_" + event['turretTier']
                timestep_data['events'].append((schema, [event[key] for key in format['events'][schema]]))
            # Store the following events in timestep_data['events'] but process them
            if schema == 'ward_placed':
                ward_type = event['wardType']
                if ward_type != 'unknown':
                    placer_index = event['placer'] - 1
                    limit_ward_type = WARD_TYPE_MAP[ward_type]
                    # Calculate ward expiration
                    lifetime = WARD_DURATIONS[ward_type]
                    if ward_type == 'yellowTrinket':
                        level_index = format['state']['champion_state'].index('level')
                        average_level = np.average([timestep_data['state']['champion_state'][i][level_index]
                                                   for i in range(10)])
                        lifetime = lifetime(average_level)
                    ward_state = [event['position']['x'], event['position']['z']]
                    if lifetime != np.inf:
                        lifetime -= (current_timestep - event['gameTime'])  # Remaining lifetime on the next timestamp
                        ward_state.append(lifetime)
                    timestep_data['state'][f'{limit_ward_type}_wards'][placer_index] = [ward_state] + \
                                                                                       timestep_data['state'][f'{limit_ward_type}_wards'][placer_index][:-1]
                    placer_limit = WARD_LIMITS[limit_ward_type]
                    if [item for item in timestep_data['state']['items'][placer_index]
                        if item[format['state']['items'].index('itemID')] == vigilant_wardstone_id]:
                        placer_limit += 1
                    for i in range(placer_limit, len(timestep_data['state'][f'{limit_ward_type}_wards'][placer_index])):
                        timestep_data['state'][f'{limit_ward_type}_wards'][placer_index][i] = \
                            [0 for _ in range(len(format['state'][f'{limit_ward_type}_wards']))]
            elif schema == 'ward_killed':
                ward_type = event['wardType']
                if ward_type != 'unknown':
                    ward_kill_attempts += 1
                    possible_placers = range(5) if event['killer'] > 5 else range(5, 10)
                    limit_ward_type = WARD_TYPE_MAP[ward_type]
                    ward_removed = False
                    for possible_placer in possible_placers:
                        for ward in timestep_data['state'][f'{limit_ward_type}_wards'][possible_placer]:
                            if ward[0] == event['position']['x'] and ward[1] == event['position']['z']:
                                timestep_data['state'][f'{limit_ward_type}_wards'][possible_placer].remove(ward)
                                timestep_data['state'][f'{limit_ward_type}_wards'][possible_placer].append(
                                    [0 for _ in range(len(format['state'][f'{limit_ward_type}_wards']))])
                                ward_removed = True
                                break
                        if ward_removed:
                            ward_kill_successes += 1
                            break
            elif schema == 'building_destroyed' and event['buildingType'] == 'inhibitor':
                team_index = event['teamID'] // 100 - 1
                inhibitor_destroyed_times[team_index, LANES.index(event['lane'])] = event['gameTime']
            elif schema == 'epic_monster_kill':
                monster_type = event['monsterType']
                killer_team = event['killerTeamID'] // 100 - 1
                if monster_type in NEUTRAL_EPIC_MONSTERS:
                    monster_killed_times[monster_respawns.index(monster_type)] = event['gameTime']
                    if monster_type == 'baron':
                        baron_killed_times[killer_team] = event['gameTime']
                    elif monster_type == 'dragon':
                        if dragon_type == 'elder':
                            elder_dragon_killed_times[killer_team] = event['gameTime']
                        else:
                            timestep_data['state']['team_buffs'][killer_team][team_buff_state.index(dragon_type)] += 1
                            dragons_killed[killer_team] += 1
                            # Check if elder dragon spawns next
                            if dragons_killed.max() >= 4:
                                monster_respawn_vector[monster_respawns.index(monster_type)] = 6 * 60 * 1000
                        dragon_type = None
                    elif monster_type == 'scuttleCrab':
                        # Scuttle crabs start respawning after the 2 small ones get killed
                        crabs_killed += 1
                        if crabs_killed < 2:
                            monster_killed_times[monster_respawns.index(monster_type)] = -np.inf
                else:
                    monster_jungle_team = killer_team if not event['inEnemyJungle'] else 1 - killer_team
                    monster_killed_times[monster_respawns.index(f't{monster_jungle_team + 1}_{monster_type}')] = event['gameTime']
                    if monster_type in ('redCamp', 'blueCamp'):
                        buff_acquired_times[event['killer'] - 1, buff_state.index(f'{monster_type[:-4]}_remaining')] = event['gameTime']
            elif schema == 'epic_monster_spawn':
                dragon_type = event['dragonType']
            elif schema == 'champion_kill':
                buffs_stolen = buff_acquired_times[event['victim'] - 1] + CAMP_BUFF_DURATIONS > event['gameTime']
                buff_acquired_times[event['killer'] - 1, buffs_stolen] = event['gameTime']
                buff_acquired_times[event['victim'] - 1] = -np.inf

        # Append the last game ending timestep
        if 'team_stats' in timestep_data['state']:  # Only save state if we have seen a stats_update event
            update_timestep_data(timestep_data,
                                 current_timestep,
                                 inhibitor_destroyed_times,
                                 monster_killed_times,
                                 monster_respawn_vector,
                                 monster_spawn_vector,
                                 buff_acquired_times,
                                 baron_killed_times,
                                 elder_dragon_killed_times,
                                 team_buff_state,
                                 monster_respawns)
            data['data'].append(copy.deepcopy(timestep_data))

        if data['data']:
            # Save the aggregated file
            gameid = events[0]['gameID']
            with open(os.path.join(output_folder, f'{gameid}.json'), 'w') as f:
                json.dump(data, f)
    print(f"Ward removal success rate {ward_kill_successes / ward_kill_attempts * 100.0:.2f}%")


if __name__ == '__main__':
    main()
