from enum import Enum

class Roles(Enum):
    top = '0'
    jungle = '1'
    mid = '2'
    marksman = '3'
    support = '4'

class FOWKeys(Enum):
    players = 'players'
    wards = 'wards'
    buildings = 'buildings'
    minions = 'minions'

class PlayerKeys(Enum):
    champion_name = 'championName'
    player_name = 'playerName'

class TeamKeys(Enum):
    t1 = 'teamOne'
    t2 = 'teamTwo'
    
class Lanes(Enum):
    top = 'top'
    mid = 'mid'
    bot = 'bot'
    
class TurretTier(Enum):
    nexus_top = 'nexus_top'
    nexus_bot = 'nexus_bot'
    turret_base = 'turret_base'
    turret_inner = 'turret_inner'
    turret_outer = 'turret_outer'

class Builiding(Enum):
    inhibitor = 'inhibitor'
    turret = 'turret'

WAYPOINT_TOP_CURVE = [[1655, 11980],[2413, 13057],[3249, 13555]]
WAYPOINT_BOT_CURVE= [[12321, 1485],[13295, 2167],[13831, 3068]]

# TEAM 1 
#TOP

# center point is defined as midle point between nexus towers
CENTER_POINT_TEAM1 = [12831.5,12848.0]

TURRET_TOP_BASE_TEAM1 = [10481, 13650]
TURRET_TOP_INNER_TEAM1 =[7943, 13411]
TURRET_TOP_OUTER_TEAM1 = [4318, 13875]

# MID
TURRET_MID_BASE_TEAM1 = [11134, 11207]
TURRET_MID_INNER_TEAM1 = [9767, 10113]
TURRET_MID_OUTER_TEAM1 = [8955, 8510]
TURRET_MID_BOT_NEXUS_TEAM1 = [13052, 12612]
TURRET_MID_TOP_NEXUS_TEAM1 = [12611, 13084]

# BOTTOM
TURRET_BOT_BASE_TEAM1 = [13624, 10572]
TURRET_BOT_INNER_TEAM1 = [13327, 8226]
TURRET_BOT_OUTER_TEAM1 = [13866, 4505]

# INHIBITORS
INHIBITOR_TOP_TEAM1 = [11261, 13676]
INHIBITOR_MID_TEAM1 = [11603, 11667]
INHIBITOR_BOT_TEAM1 = [13604, 11316]

TEAM1_BUILDINGS_DICT = {Lanes.top.name:{TurretTier.turret_base.name:TURRET_TOP_BASE_TEAM1,TurretTier.turret_inner.name:TURRET_TOP_INNER_TEAM1,TurretTier.turret_outer.name:TURRET_TOP_OUTER_TEAM1},
                   Lanes.mid.name:{TurretTier.turret_base.name:TURRET_MID_BASE_TEAM1,TurretTier.turret_inner.name:TURRET_MID_INNER_TEAM1,TurretTier.turret_outer.name:TURRET_MID_OUTER_TEAM1,  
                                   TurretTier.nexus_bot.name:TURRET_MID_BOT_NEXUS_TEAM1,TurretTier.nexus_top.name:TURRET_MID_TOP_NEXUS_TEAM1},
                   Lanes.bot.name:{TurretTier.turret_base.name:TURRET_BOT_BASE_TEAM1,TurretTier.turret_inner.name:TURRET_BOT_INNER_TEAM1,TurretTier.turret_outer.name:TURRET_BOT_OUTER_TEAM1},
                   'inhibitor':{Lanes.top.name:INHIBITOR_TOP_TEAM1,Lanes.mid.name:INHIBITOR_MID_TEAM1,Lanes.bot.name:INHIBITOR_BOT_TEAM1}}


TEAM1_TOWERS = [TURRET_TOP_BASE_TEAM1,TURRET_TOP_INNER_TEAM1,TURRET_TOP_OUTER_TEAM1,
                   TURRET_MID_BASE_TEAM1,TURRET_MID_INNER_TEAM1,TURRET_MID_OUTER_TEAM1,TURRET_MID_BOT_NEXUS_TEAM1,TURRET_MID_TOP_NEXUS_TEAM1,
                   TURRET_BOT_BASE_TEAM1,TURRET_BOT_INNER_TEAM1,TURRET_BOT_OUTER_TEAM1]

TEAM1_INHIBITORS = [INHIBITOR_TOP_TEAM1,INHIBITOR_MID_TEAM1,INHIBITOR_BOT_TEAM1]

# TEAM 2

# center point is defined as midle point between nexus towers
CENTER_POINT_TEAM2 =[1962.5,2038.5]

#TOP
TURRET_TOP_BASE_TEAM2 = [1169, 4287]
TURRET_TOP_INNER_TEAM2 =[1512, 6699]
TURRET_TOP_OUTER_TEAM2 = [981, 10441]

# MID
TURRET_MID_BASE_TEAM2 = [3651, 3696]
TURRET_MID_INNER_TEAM2 = [5048, 4812]
TURRET_MID_OUTER_TEAM2 = [5846, 6396]
TURRET_MID_BOT_NEXUS_TEAM2 = [2177, 1807]
TURRET_MID_TOP_NEXUS_TEAM2 = [1748, 2270]

# BOTTOM
TURRET_BOT_BASE_TEAM2 = [4281, 1253]
TURRET_BOT_INNER_TEAM2 = [6919, 1483]
TURRET_BOT_OUTER_TEAM2 = [10504, 1029]

# INHIBITORS
INHIBITOR_TOP_TEAM2 = [1171, 3571]
INHIBITOR_MID_TEAM2 = [3203, 3208]
INHIBITOR_BOT_TEAM2 = [3452, 1236]

TEAM2_BUILDINGS_DICT = {Lanes.top.name:{TurretTier.turret_base.name:TURRET_TOP_BASE_TEAM2,TurretTier.turret_inner.name:TURRET_TOP_INNER_TEAM2,TurretTier.turret_outer.name:TURRET_TOP_OUTER_TEAM2},
                   Lanes.mid.name:{TurretTier.turret_base.name:TURRET_MID_BASE_TEAM2,TurretTier.turret_inner.name:TURRET_MID_INNER_TEAM2,TurretTier.turret_outer.name:TURRET_MID_OUTER_TEAM2,  
                                   TurretTier.nexus_bot.name:TURRET_MID_BOT_NEXUS_TEAM2,TurretTier.nexus_top.name:TURRET_MID_TOP_NEXUS_TEAM2},
                   Lanes.bot.name:{TurretTier.turret_base.name:TURRET_BOT_BASE_TEAM2,TurretTier.turret_inner.name:TURRET_BOT_INNER_TEAM2,TurretTier.turret_outer.name:TURRET_BOT_OUTER_TEAM2},
                   'inhibitor':{Lanes.top.name:INHIBITOR_TOP_TEAM2,Lanes.mid.name:INHIBITOR_MID_TEAM2,Lanes.bot.name:INHIBITOR_BOT_TEAM2}}



TEAM2_TOWERS = [TURRET_TOP_BASE_TEAM2,TURRET_TOP_INNER_TEAM2,TURRET_TOP_OUTER_TEAM2,
                   TURRET_MID_BASE_TEAM2,TURRET_MID_INNER_TEAM2,TURRET_MID_OUTER_TEAM2,TURRET_MID_BOT_NEXUS_TEAM2,TURRET_MID_TOP_NEXUS_TEAM2,
                   TURRET_BOT_BASE_TEAM2,TURRET_BOT_INNER_TEAM2,TURRET_BOT_OUTER_TEAM2]

TEAM2_INHIBITORS = [INHIBITOR_TOP_TEAM2,INHIBITOR_MID_TEAM2,INHIBITOR_BOT_TEAM2]


INHIBITOR_RESPAWN_TIME = 300
MINION_BASE_MOVEMENT_SPEED = 325

# calculated by get_waypoints from: minion_waypoints.py
MINION_WAYPOINTS = {'teamTwo': {
                        'top': [[12831.5, 12848.0], [10481, 13650], [7943, 13411], [4318, 13875], [3249, 13555], [2413, 13057], [1655, 11980], [981, 10441], [1512, 6699], [1169, 4287], [1962.5, 2038.5]], 
                        'mid': [[12831.5, 12848.0], [11134, 11207], [9767, 10113], [8955, 8510], [5846, 6396], [5048, 4812], [3651, 3696], [1962.5, 2038.5]], 
                        'bot': [[12831.5, 12848.0], [13624, 10572], [13327, 8226], [13866, 4505], [13831, 3068], [13295, 2167], [12321, 1485], [10504, 1029], [6919, 1483], [4281, 1253], [1962.5, 2038.5]]}, 
                    'teamOne': {
                        'top': [[1962.5, 2038.5], [1169, 4287], [1512, 6699], [981, 10441], [1655, 11980], [2413, 13057], [3249, 13555], [4318, 13875], [7943, 13411], [10481, 13650], [12831.5, 12848.0]], 
                        'mid': [[1962.5, 2038.5], [3651, 3696], [5048, 4812], [5846, 6396], [8955, 8510], [9767, 10113], [11134, 11207], [12831.5, 12848.0]], 
                        'bot': [[1962.5, 2038.5], [4281, 1253], [6919, 1483], [10504, 1029], [12321, 1485], [13295, 2167], [13831, 3068], [13866, 4505], [13327, 8226], [13624, 10572], [12831.5, 12848.0]]}}


#NOT EXACT LOCATIONS
BUSHES = [[8041,936],[12895,1467],[13278,1930],[9373,2031],[13718,2347],
          [10502,3103],[6935,3013],[12240,3408],[8278,3442],[5626,3566],
          [6721,4706],[12691,5169],[12048,5598],[9825,5688],[10095,6444],
          [8098,6918],[14125,7031],[11664,7110],[4858,7200],[3402,7878],
          [10141,8081],[6834,8148],[852,7990],[5028,8634],[5152,9345],
          [3075,9774],[2375,9943],[8323,10316],[6213,10395],[9351,11444],
          [6856,11681],[8053,12054],[4475,11963],[2770,11817],[5727,12934],
          [1258,12663],[1676,12990],[2003,13498],[7229,14085],[8848,4673]]


exclude_keys = [
    'ability3Name', 'ability2Level', 'accountID', 'teamID', 'ability2Name',
    'ability1Name', 'ability1Level', 'ability4Level', 'XPForNextLevel',
    'championName', 'ultimateName', 'playerName', 'participantID',
    'ability4Name', 'ability3Level', 'summonerSpell1CooldownRemaining',
    'ability4CooldownRemaining', 'summonerSpell2CooldownRemaining',
    'ability3CooldownRemaining', 'summonerSpell2Name', 'ability1CooldownRemaining',
    'position', 'alive', 'summonerSpell1Name', 'respawnTimer', 'ability2CooldownRemaining',
    'ultimateCooldownRemaining','items','stats'
]


selected_stats_keys = [
    'MINIONS_KILLED', 'NEUTRAL_MINIONS_KILLED', 'NEUTRAL_MINIONS_KILLED_YOUR_JUNGLE',
    'NEUTRAL_MINIONS_KILLED_ENEMY_JUNGLE', 'CHAMPIONS_KILLED', 'NUM_DEATHS', 'ASSISTS',
    'WARD_PLACED', 'WARD_KILLED', 'VISION_SCORE', 'TOTAL_DAMAGE_DEALT',
    'PHYSICAL_DAMAGE_DEALT_PLAYER', 'MAGIC_DAMAGE_DEALT_PLAYER',
    'TRUE_DAMAGE_DEALT_PLAYER', 'TOTAL_DAMAGE_DEALT_TO_CHAMPIONS',
    'TOTAL_DAMAGE_TAKEN', 'PHYSICAL_DAMAGE_TAKEN', 'MAGIC_DAMAGE_TAKEN',
    'TRUE_DAMAGE_TAKEN', 'TOTAL_DAMAGE_SELF_MITIGATED',
    'TOTAL_DAMAGE_SHIELDED_ON_TEAMMATES', 'TOTAL_DAMAGE_DEALT_TO_BUILDINGS',
    'TOTAL_DAMAGE_DEALT_TO_TURRETS', 'TOTAL_DAMAGE_DEALT_TO_OBJECTIVES',
    'TOTAL_TIME_CROWD_CONTROL_DEALT', 'TOTAL_HEAL_ON_TEAMMATES', 'TIME_CCING_OTHERS'
]
