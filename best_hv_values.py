# Code by @maiemile

from pathlib import Path

best_hvs = {}

with open('indicator_data\\indicator_values_log.txt', 'r') as file:
    for line in file:
        split_line = line.split() 
        config = split_line[0].split('-')
        if config[1][0] == '9':
            continue
        problem = config[0] + '-' + config[1]
        configuration = config[2] + '-' + config[3] + '-' + config[4]
        try:
            hv = float(split_line[1])
        except:
            continue # Error
        try:
            if hv > best_hvs[problem][0]:
                best_hvs[problem] = (hv, configuration)
        except:
            best_hvs[problem] = (hv, configuration)

# This next loop is just for checking that there are no other configurations that have achieved the same HV value
with open('indicator_data\\indicator_values_log.txt', 'r') as file:
    for line in file:
        split_line = line.split() 
        config = split_line[0].split('-')
        if config[1][0] == '9':
            continue
        problem = config[0] + '-' + config[1]
        configuration = config[2] + '-' + config[3] + '-' + config[4]
        try:
            hv = float(split_line[1])
        except:
            continue # Error

file_name = Path('indicator_data\\best_hv_values.txt')

with open(file_name, "w") as file:
    for k,v in best_hvs.items():
        file.write(" ".join([k, v[1]]) + "\n")
