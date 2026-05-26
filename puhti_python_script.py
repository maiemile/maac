# code by @maiemile

# quick setup for the baseline path
BASE_PATH = ''

import random
from multiprocessing import Pool, cpu_count
from pathlib import Path
import logging

import numpy as np

from desdeo.problem import Objective, Problem, Simulator, Variable
from desdeo.emo.hooks.archivers import NonDominatedArchive
from desdeo.emo.methods.templates import template1
from desdeo.emo.operators.evaluator import EMOEvaluator
from desdeo.emo.operators.mutation import (BoundedPolynomialMutation, NonUniformMutation, MPTMutation,
                                            PowerMutation)
from desdeo.emo.operators.crossover import (SimulatedBinaryCrossover, BlendAlphaCrossover, 
                                            SingleArithmeticCrossover, LocalCrossover)
from desdeo.emo.operators.termination import MaxEvaluationsTerminator
from desdeo.tools.patterns import Publisher
from desdeo.emo.operators.selection import (
    NSGAIII_select,
    RVEASelector,
    ReferenceVectorOptions,
    ParameterAdaptationStrategy,
    IBEA_Selector
)
from desdeo.emo.operators.generator import RandomGenerator
from desdeo.problem.schema import (
    Objective,
    Problem,
    Simulator,
    Variable,
)

import reproblem as reprob
from desdeo.tools.utils import repair


# dictionaries to fetch classes of operators from DESDEO or problems from separate files
algorithms = {"nsga3": NSGAIII_select, "rvea": RVEASelector, "ibea": IBEA_Selector}
crossovers = {"SBX": SimulatedBinaryCrossover, "Balpha": BlendAlphaCrossover, "Single": SingleArithmeticCrossover, 
              "Local": LocalCrossover}
mutations = {"BPM": BoundedPolynomialMutation, "MPTM": MPTMutation, "NUM": NonUniformMutation, "PM": PowerMutation}
re_problems = {"re31": reprob.RE31, "re32": reprob.RE32, "re33": reprob.RE33, "re34": reprob.RE34, "re37": reprob.RE37,
               "re41": reprob.RE41, "re42": reprob.RE42, "re61": reprob.RE61, "re91": reprob.RE91}
pop_sizes = {3: 105, 4: 120, 6: 132, 9: 210} # from the RVEA article, partially interpolated

_seed = 1
f_evaluations = 10000

logging.basicConfig(filename='indicator_values_final_pop_v3.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def get_experiments():

    # all the problems instances
    problem_instances = [
        # DTLZ problems (3, 4, 6 and 9 objectives, 4 * 7 = 28 instances)
        ["dtlz1", 7, 3],["dtlz2", 10, 3],["dtlz3", 10, 3],["dtlz4", 10, 3],["dtlz5", 10, 3],["dtlz6", 10, 3],["dtlz7", 10, 3],
        ["dtlz1", 11, 4],["dtlz2", 15, 4],["dtlz3", 15, 4],["dtlz4", 15, 4],["dtlz5", 15, 4],["dtlz6", 15, 4],["dtlz7", 15, 4], #DTLZ5-7 with 4 objectives doesn't work, "not implemented yet" in pymoo
        ["dtlz1", 11, 6],["dtlz2", 15, 6],["dtlz3", 15, 6],["dtlz4", 15, 6],["dtlz5", 15, 6],["dtlz6", 15, 6],["dtlz7", 15, 6],
        ["dtlz1", 11, 9],["dtlz2", 15, 9],["dtlz3", 15, 9],["dtlz4", 15, 9],["dtlz5", 15, 9],["dtlz6", 15, 9],["dtlz7", 15, 9],

        # WFG problems (3, 4, 6 and 9 objectives, 4 * 9 = 36 instances)
        ["wfg1", 10, 3],["wfg2", 10, 3],["wfg3", 10, 3],["wfg4", 10, 3],["wfg5", 10, 3],["wfg6", 10, 3],["wfg7", 10, 3],["wfg8", 10, 3],["wfg9", 10, 3],
        ["wfg1", 15, 4],["wfg2", 14, 4], ["wfg3", 14, 4],["wfg4", 15, 4],["wfg5", 15, 4],["wfg6", 15, 4],["wfg7", 15, 4],["wfg8", 15, 4],["wfg9", 15, 4], #WFG2 and WFG3 require an even number of decision variables
        ["wfg1", 15, 6],["wfg2", 14, 6], ["wfg3", 14, 6],["wfg4", 15, 6],["wfg5", 15, 6],["wfg6", 15, 6],["wfg7", 15, 6],["wfg8", 15, 6],["wfg9", 15, 6],
        ["wfg1", 18, 9],["wfg2", 18, 9], ["wfg3", 18, 9],["wfg4", 18, 9],["wfg5", 18, 9],["wfg6", 18, 9],["wfg7", 18, 9],["wfg8", 18, 9],["wfg9", 18, 9],

        # RE problems, 8 instances
        ["re31", 3, 3],
        ["re32", 4, 3],
        ["re33", 4, 3],
        ["re34", 5, 3],
        ["re37", 4, 3],
        #["re41", 7, 4], # not included due to unexpected results while calculating the approximate Pareto front
        #["re42", 6, 4],
        #["re61", 3, 6], 
    ]

    algos = ["nsga3", "rvea", "ibea"]
    cxs = ["SBX", "Balpha", "Single", "Local"]
    mxs = ["BPM", "MPTM", "NUM", "PM"]

    #####################################################3

    # Enumerate all possible problem instance + config permutations
    prob_and_config_permutations = [prob + [a, cx, mx] for prob in problem_instances for a in algos for cx in cxs for mx in mxs]

    # Randomize the order
    random.seed(1)
    random.shuffle(prob_and_config_permutations)
    
    return prob_and_config_permutations

def simulator_problem(problem_name: str, n_vars: int, n_objs: int, server=False) -> Problem:
    # define all the variables separately in DESDEO
    file = "pymoo_simulator.py"
    if server:
        file = r"http://127.0.0.1:8000/evaluate"

    if problem_name[:3] == 'wfg':
        variables = [
        Variable(name=f"x_{i+1}", symbol=f"x_{i+1}", variable_type="real", lowerbound=0, upperbound=2*(i+1))
        for i in range(n_vars)
        ]
        
    if problem_name[:2] == 're':
        problem_class = re_problems[problem_name]
        lowerbounds = problem_class().lbound
        upperbounds = problem_class().ubound
        if server:
            file = r"http://127.0.0.1:8000/evaluate_re"
        variables = [
            Variable(name=f"x_{i+1}", symbol=f"x_{i+1}", variable_type="real", lowerbound=lowerbounds[i], upperbound=upperbounds[i])
            for i in range(n_vars)
        ]
    if problem_name[:4] == 'dtlz':
        variables = [
            Variable(name=f"x_{i+1}", symbol=f"x_{i+1}", variable_type="real", lowerbound=0, upperbound=1)
            for i in range(n_vars)
        ]

    # define all objectives for DESDEO based on the number of objective funcitions
    objectives = [
        Objective(
            name=f"f_{i+1}",
            symbol=f"f_{i+1}",
            simulator_path=Path(file),  # simulator file
            objective_type="simulator",
        )
        for i in range(n_objs)
    ]

    return Problem(
        name="Simulator problem",
        description="",
        variables=variables,
        objectives=objectives,
        simulators=[
            Simulator(
                name="s_1",
                symbol="s_1",
                file=Path(file),
                parameter_options={
                    "name": problem_name,
                    "n_vars": n_vars,
                    "n_objs": n_objs,
                },
            )
        ],
    )


def run_experiment(prob_name, n_vars, n_objs, algo, cx, mx):
    import requests
    import time
    attempts = 0

    while attempts < 20:
        try:
            r = requests.get('http://127.0.0.1:8000/test')
            r.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xxx
            break
        except:
            attempts += 1
            time.sleep(10)

    # create a unique problem name
    prob_name_print = prob_name + '-' + str(n_objs) + 'obj'
    configuration = "-" + algo + "-" + cx + "-" + mx

    # create a DESDEO problem object for simulator-based problems 
    prob = simulator_problem(prob_name, n_vars, n_objs, True)

    pop_size = pop_sizes[n_objs]

    try:
        algorithm_s = algorithms[algo]
        crossover_s = crossovers[cx]
        mutation_s = mutations[mx]
        publisher = Publisher()
        # EMOEvaluator is used to evaluate the solutions
        evaluator = EMOEvaluator(
            problem=prob,
            publisher=publisher,
            verbosity=2
        )
        crossover = crossover_s(
            problem=prob,
            publisher=publisher,
            seed=_seed,
            verbosity=1
        )
        if mx == "NUM":
            mutation = mutation_s(
                problem=prob,
                publisher=publisher,
                verbosity=1,
                seed=_seed,
                max_generations = int(f_evaluations/pop_size)
            )
        else:
            mutation = mutation_s(
                problem=prob,
                publisher=publisher,
                verbosity=1,
                seed=_seed,
            )
        reference_vector_options = ReferenceVectorOptions(
            number_of_vectors=pop_size,
        )
        if algo == "nsga3":
            selector = algorithm_s(
                problem=prob,
                publisher=publisher,
                reference_vector_options=reference_vector_options,
                verbosity=2,
            )
        if algo == "rvea":
            selector = algorithm_s(
                problem=prob,
                publisher=publisher,
                reference_vector_options=reference_vector_options,
                verbosity=2,
                parameter_adaptation_strategy=ParameterAdaptationStrategy.FUNCTION_EVALUATION_BASED
            )
        if algo == "ibea":
            selector = algorithm_s(
                problem=prob,
                verbosity=2,
                publisher=publisher,
                population_size=pop_size,
            )
        # generate the initial population randomly
        generator = RandomGenerator(
            problem=prob,
            evaluator=evaluator,
            publisher=publisher,
            n_points=pop_size,
            verbosity=2,
            seed=_seed
        )
        terminator = MaxEvaluationsTerminator(f_evaluations, publisher=publisher)
        archive = NonDominatedArchive(problem=prob, publisher=publisher)
        # Register the components to the publisher
        components = [evaluator, generator, crossover, mutation, selector, terminator, archive]
        [publisher.auto_subscribe(x) for x in components]
        [publisher.register_topics(x.provided_topics[x.verbosity], x.__class__.__name__) for x in components]
        consistency_check = publisher.check_consistency()
        # make sure the verbosity levels have been set on correct levels
        if consistency_check[0] == False:
            return
        # makes sure the variables stay within the set bounds
        repair_func = repair(
            lower_bounds={v.symbol: v.lowerbound for v in prob.get_flattened_variables()},
            upper_bounds={v.symbol: v.upperbound for v in prob.get_flattened_variables()},
        )
        res = template1(
            crossover=crossover,
            mutation=mutation,
            selection=selector,
            generator=generator,
            terminator=terminator,
            evaluator=evaluator,
            repair=repair_func
        )

        objective_names = [obj.name for obj in prob.objectives]
        final_pop = np.array(res.outputs[objective_names])

        archived_solutions = np.array(archive.solutions[objective_names])

        # log the indicator values
        file_name = Path(BASE_PATH + '/archived_pops_v3/' + prob_name_print + configuration + '.txt')

        # save the archived population to a txt file
        with open(file_name, "w") as file:
            for line in archived_solutions:
                file.write(" ".join(str(x) for x in line.tolist()) + "\n")

        file_name_fp = Path(BASE_PATH + '/archived_final_pops_v3/' + prob_name_print + configuration + '.txt')

        with open(file_name_fp, "w") as file:
            for line in final_pop:
                file.write(" ".join(str(x) for x in line.tolist()) + "\n") 


        log_text = prob_name_print + ' ' + algo + ' ' + cx + ' ' + mx
        logger.info('%s', log_text)
    except:
        print(prob_name_print, algo, cx, mx, 999999, -999999)
        log_text = prob_name_print + ' ' + algo + ' ' + cx + ' ' + mx + " ERROR"
        logger.info('%s', log_text)


if __name__ == "__main__":

    print(cpu_count())
    # fetch the full experiment list
    experiment_list = get_experiments()

    print(len(experiment_list))
    
    from pymoo_server import start_server
    
    # Create a pool of workers and run the function processImage for each filepath in the list
    with Pool(processes=20) as pool:
        pool.apply_async(start_server)
        pool.starmap(run_experiment, experiment_list)
        pool.terminate()
        pool.join()
