import os
import json

# Define your directory path here
directory_path = 'C:\\Users\\edems\\Documents\\Work\\FOW_AI\\Game_Data\\Headers_Games'

# Get a list of all JSON files in the directory
json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]

final_dict = []
# Iterate over each file
for json_file in json_files:
    # Construct the full file path
    file_path = os.path.join(directory_path, json_file)
    
    # Open the file and load the JSON data
    with open(file_path, 'r') as file:
        data = json.load(file)
        
        # Now data contains the parsed JSON data from the file
        # Let's find the first summonerName with less than two spaces for each team
        for event in data['events']:
            first_summoner_name_t1 = None
            first_summoner_name_t2 = None
            found = False
            for team in ['teamOne', 'teamTwo']:
                if team in event:
                    if team == 'teamOne':
                        for participant in event[team]:
                            if 'summonerID' in participant and 'accountID' in participant and participant['summonerName'].count(' ') < 2:
                                first_summoner_name_t1 = participant['summonerName']
                                if first_summoner_name_t1 != None:
                                    break
                    else:
                        for participant in event[team]:
                            if 'summonerID' in participant and 'accountID' in participant and participant['summonerName'].count(' ') < 2:
                                first_summoner_name_t2 = participant['summonerName']
                                if first_summoner_name_t2 != None:
                                    break
                if first_summoner_name_t1 is not None and first_summoner_name_t2 is not None:
                    final_dict.append({'game':json_file.split()[0],'t1':first_summoner_name_t1.split()[0],'t2':first_summoner_name_t2.split()[0]})
                    found = True
                    break
            if found:
                break
            
TEAM_NAMES = ['FLY','TL']

# Create a new list to store dictionaries with valid 't1' and 't2' values
valid_dicts = []

# Iterate over each dictionary in the final_dict list
for dictionary in final_dict:
    # Check if 't1' or 't2' values are in the TEAM_NAMES list
    if dictionary['t1'] in TEAM_NAMES and dictionary['t2'] in TEAM_NAMES:
        # If so, add the dictionary to the valid_dicts list
        valid_dicts.append(dictionary)

# Replace the original list with the new one containing only valid dictionaries

# Open a new text file in write mode
with open('games_found_had.txt', 'w') as f:
    # Iterate over each dictionary in the valid_dicts list
    for dictionary in valid_dicts:
        # Write the 'game' data to the text file, followed by a newline
        f.write(dictionary['game'] + '\n')

