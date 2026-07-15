# created by @maiemile

import sqlite3
import itertools
import re

# TODO: when new parameter options are given, create new columns, unique must be updated

pattern = "^[A-Za-z0-9_-]*$"

# TODO: this should be loaded from the config file/user input
database = "maac.db"

def check_key(key:str) -> None:
    '''
    Checks the integrity of the given string. If it contains characters outside of the defined pattern,
    an exception is thrown.
    '''
    check_key = bool(re.match(pattern, key))
    if check_key == False:
        raise Exception(f'Invalid value: {key}. The value must correspond to the pattern {pattern}.')
    

def query_data(sql_statement:str, qmark:tuple=None) -> list:
    '''
    Queries data according to the provided SQL statement.
    Return the data as a list of tuples.
    '''
    data = []
    try:
        with sqlite3.connect(database) as conn:
            cur = conn.cursor()
            if qmark == None:
                cur.execute(sql_statement)
                
            #If the user has given parameters, fill them in the SQL statement
            else:
                cur.execute(sql_statement, qmark)
            rows = cur.fetchall()

            # deal with 1 row or 2<= rows of data separately
            if len(rows) == 1:
                data = rows[0]
            else:
                for row in rows:
                    data.append(row)
    except sqlite3.Error as e:
        print(e)

    return data


def create_table(sql_statement:str) -> None:
    ''' 
    Creates an SQL table based on the provided SQL statement.
    '''

    # create the specified table
    try:
        with sqlite3.connect(database) as conn:
            # create a cursor
            cursor = conn.cursor()

            # execute statement
            cursor.execute(sql_statement)

            # commit the changes
            conn.commit()

            print("Tables created successfully.")
    except sqlite3.OperationalError as e:
        print("Failed to create tables:", e)


def insert_data(sql_statement:str, data:list) -> None:
    '''
    Inserts the data into a table using the provided SQL statement.
    '''
    # add data to the table
    try:
        with sqlite3.connect(database) as conn:
            # create a cursor
            cursor = conn.cursor()

            # add each combination to the table
            for line in data:
                try:
                    cursor.execute(sql_statement, line)
                # If the combination already exists in the database, skip it
                except sqlite3.IntegrityError as e:
                    continue

            # commit the changes
            conn.commit()

            print("Data created.")
    except sqlite3.OperationalError as e:
        print("Failed to create data:", e)


def generate_ea_table(options:dict) -> None:
    '''
    Generates an SQL table of all possible configuration combinations according to the configuration options
    included in a dictionary.
    '''

    sql_statement =  """CREATE TABLE IF NOT EXISTS eas (
            ea_id INTEGER PRIMARY KEY AUTOINCREMENT, 
            name TEXT
            """
    
    # add the configuration options and their types to the sql statement
    # we also add a unique statement to avoid duplicate configurations
    unique = ",\nunique ("
    # fetch the list of options per parameter and get a list of keys
    option_lists = []
    keys = []
    for key,value in options.items():
        keys.append(key)
        data_type, values = value
        option_lists.append(values)

        # Check integrity of the key and the data type
        check_key(key)
        check_key(data_type)
        
        # add new column to SQL statement
        sql_statement += f",\n{key} {data_type}"
        unique += f"{key}, "

    unique = unique[:-2]+""")"""
    sql_statement += unique + """\n);"""

    create_table(sql_statement)
    
    # iterate the list of options to get all possible combinations
    all_combinations = list(itertools.product(*option_lists))

    # create the insert statement
    sql = f'''INSERT INTO eas('''
    for key in keys:
        sql += f'''{key},'''
    sql = sql[:-1] + ''')\nVALUES('''
    for _ in range(len(keys)):
        sql += '''?,'''
    sql = sql[:-1] + ''')'''

    insert_data(sql, all_combinations)


def generate_problem_table(problems:list[str,int,int]) -> None:
    '''
    Generates an SQL table with the problem data. 
    Data contains an automatically generated ID.
    The user must supply the name, number of objective functions and the number of variables 
    for each problem.
    '''

    table_sql = '''CREATE TABLE IF NOT EXISTS problems (
            problem_id INTEGER PRIMARY KEY AUTOINCREMENT, 
            name TEXT NOT NULL,
            obj INTEGER NOT NULL,
            var INTEGER NOT NULL,
            unique (name, obj, var)
            );'''
    
    create_table(table_sql)

    # create the insert statement
    sql = f'''INSERT INTO problems(name,obj,var)
        VALUES(?,?,?)'''

    insert_data(sql, problems)


def generate_run_table(n_of_repeats:list[int]=[1], target_evals:list[int]=[10000]) -> None:
    '''
    Generates an SQL table of all algorithm configuration runs.
    A separate row is created for each configuration + problem pair. 
    These rows are repeated for each value in target_evals as many times as the value of n_of_repeats in the same index.
    '''

    if len(n_of_repeats) != len(target_evals):
        raise Exception('The length of n_of_repeats and target_evals must be equal')
    # TODO: exception if n_of_repeats < 1 or target_evals < 1
    # TODO: add indicator columns
    table_sql = '''CREATE TABLE IF NOT EXISTS runs(
            run_id INTEGER PRIMARY KEY AUTOINCREMENT, 
            ea_id INTEGER NOT NULL,
            problem_id INTEGER NOT NULL,
            seed INTEGER NOT NULL,
            target_evals INTEGER NOT NULL,
            unique (ea_id, problem_id, seed, target_evals),
            FOREIGN KEY(ea_id) REFERENCES eas(ea_id),
            FOREIGN KEY(problem_id) REFERENCES problems(problem_id)
            );'''
    
    create_table(table_sql)

    # create the insert statement
    sql_ea = f'''SELECT ea_id FROM eas'''
    sql_prob = f'''SELECT problem_id FROM problems'''

    ea_ids = query_data(sql_ea)
    prob_ids = query_data(sql_prob)

    ea_ids_list = [x[0] for x in ea_ids]
    prob_ids_list = [x[0] for x in prob_ids]

    # for each value of target_evals, create as many repeats as listed in n_of_repeats
    # of the same EA configuration + problem pair
    for i in range(len(n_of_repeats)):
        repeats = n_of_repeats[i]
        for j in range(repeats):
            full_data = [ea_ids_list, prob_ids_list, [j+1], [target_evals[i]]]
            all_combinations = list(itertools.product(*full_data))

            # create the insert statement
            sql_statement = f'''INSERT INTO runs(ea_id,problem_id,seed,target_evals)
                VALUES(?,?,?,?)'''
            insert_data(sql_statement, all_combinations)

    
# TODO: this dictionary should be given by the user/read from json files
options = {"selection": ["TEXT", ["IBEA", "NSGA-III", "RVEA"]], "crossover": ["TEXT", ["SBX", "SAX"]], "mutation": ["TEXT", ["BPM", "NUM"]]}
problems = [
        ["dtlz1", 3, 7],["dtlz2", 3, 10],["dtlz3", 3, 10],["dtlz4", 3, 10],["dtlz5", 3, 10],["dtlz6", 3, 10],["dtlz7", 3, 10]]


def do(n_of_repeats:list[int], target_evals:list[int]) -> None:
    '''
    Main function for generating and populating 3 SQL tables in a database:
    1) EA table 2) problem table 3) run table
    '''
    generate_ea_table(options)
    generate_problem_table(problems)
    generate_run_table(n_of_repeats=n_of_repeats, target_evals=target_evals)