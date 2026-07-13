# code by @maiemile

# quick setup for the baseline path
BASE_PATH = ''

import random
from multiprocessing import Pool, cpu_count
from pathlib import Path
import logging
import utils
import requests
import time
from pymoo_server import start_server
import numpy as np

from desdeo.problem import Objective, Problem, Simulator, Variable, Url
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
    NSGA3Selector,
    RVEASelector,
    ReferenceVectorOptions,
    ParameterAdaptationStrategy,
    IBEASelector
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
algorithms = {"nsga3": NSGA3Selector, "rvea": RVEASelector, "ibea": IBEASelector}
crossovers = {"SBX": SimulatedBinaryCrossover, "Balpha": BlendAlphaCrossover, "Single": SingleArithmeticCrossover, 
              "Local": LocalCrossover}
mutations = {"BPM": BoundedPolynomialMutation, "MPTM": MPTMutation, "NUM": NonUniformMutation, "PM": PowerMutation}
re_problems = {"re31": reprob.RE31, "re32": reprob.RE32, "re33": reprob.RE33, "re34": reprob.RE34, "re37": reprob.RE37,
               "re41": reprob.RE41, "re42": reprob.RE42, "re61": reprob.RE61, "re91": reprob.RE91}
pop_sizes = utils.get_default_pop_sizes() # from the RVEA article, partially interpolated

_seed = 1
f_evaluations = 10000

logging.basicConfig(filename='indicator_values_final_pop.log', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_experiments() -> list[list]:

    # all the problems instances
    problem_instances = utils.get_problem_instances()
    algos, cxs, mxs = utils.get_all_configuration_options()

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

    if server:
        url = Url(url=file)

    return Problem(
        name="Simulator problem",
        description="",
        variables=variables,
        objectives=objectives,
        simulators=[
            Simulator(
                name="s_1",
                symbol="s_1",
                url=url,
                #file=Path(file),
                parameter_options={
                    "name": problem_name,
                    "n_vars": n_vars,
                    "n_objs": n_objs,
                },
            )
        ],
    )


def run_experiment(prob_name: str, n_vars: int, n_objs: int, algo: str, cx: str, mx:str) -> None:
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
    # TODO: replace problem data and config data with their ids from the corresponding tables?
    prob_name_print = prob_name + '-' + str(n_objs) + 'obj' #+ '-' + str(n_vars) + 'var-' + str(f_evaluations) + 'eval-' + str(_seed) 
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
        final_pop = np.array(res.optimal_outputs[objective_names])
        archived_solutions = np.array(archive.solutions[objective_names])
        # log the indicator values
        file_name = Path(BASE_PATH + 'archived_pops/' + prob_name_print + configuration + '.txt')
        # save the archived population to a txt file
        with open(file_name, "w") as file:
            for line in archived_solutions:
                file.write(" ".join(str(x) for x in line.tolist()) + "\n")
        file_name_fp = Path(BASE_PATH + 'archived_final_pops/' + prob_name_print + configuration + '.txt')
        with open(file_name_fp, "w") as file:
            for line in final_pop:
                file.write(" ".join(str(x) for x in line.tolist()) + "\n") 
        log_text = prob_name_print + ' ' + algo + ' ' + cx + ' ' + mx
        logger.info('%s', log_text)
    except:
        print(prob_name_print, algo, cx, mx, 999999, -999999)
        log_text = prob_name_print + ' ' + algo + ' ' + cx + ' ' + mx + " ERROR"
        logger.info('%s', log_text)


def do(num_of_repeats: int = 1, function_evaluations: int = 10000) -> None:
    print(cpu_count())
    # fetch the full experiment list
    # TODO: use the "num_of_repeats" to set the number of seeds required,
    # might be more sensible to do using the database
    experiment_list = get_experiments()

    f_evaluations = function_evaluations

    # TODO: check which experiments have been completed and remove them from the list

    print(len(experiment_list))
    
    # Create a pool of workers and run the function processImage for each filepath in the list
    with Pool(processes=20) as pool:
        pool.apply_async(start_server)
        pool.starmap(run_experiment, experiment_list)
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    experiment_list = get_experiments()
    for experiment in experiment_list:
        prob_name, n_vars, n_obj, algo, cx, mx = experiment
        run_experiment(prob_name, n_vars, n_obj, algo, cx, mx)
