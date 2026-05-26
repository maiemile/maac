# Code by @maiemile

from pathlib import Path

best_igds = {}
best_igd_pluses = {}

file_name = 'indicator_data\\igd_values_log.txt'

with open(file_name, 'r') as file:
    for line in file:
        split_line = line.split() 
        config = split_line[0].split('-')
        problem = config[0] + '-' + config[1]
        configuration = config[2] + '-' + config[3] + '-' + config[4]
        try:
            igd = float(split_line[1])
            igd_plus = float(split_line[2])
        except:
            continue # Error
        try:
            if igd < best_igds[problem][0]:
                best_igds[problem] = (igd, configuration)
        except:
            best_igds[problem] = (igd, configuration)
        try:
            if igd_plus < best_igd_pluses[problem][0]:
                best_igd_pluses[problem] = (igd_plus, configuration)
        except:
            best_igd_pluses[problem] = (igd_plus, configuration)

file_name_igd = Path('indicator_data\\best_regular_igd_values2.txt')
file_name_igd_plus = Path('indicator_data\\best_igd_plus_values2.txt')

with open(file_name_igd, "w") as file:
    for k,v in best_igds.items():
        file.write(" ".join([k, v[1]]) + "\n")

with open(file_name_igd_plus, "w") as file:
    for k,v in best_igd_pluses.items():
        file.write(" ".join([k, v[1]]) + "\n") 